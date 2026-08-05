import torch
import wandb
from liger_kernel.chunked_loss import LigerFusedLinearSimPOLoss
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from transformers import Trainer

from vyvonext.core.data import AlternatingDistributedSampler
from vyvonext.core.tokens import AUDIO_OFFSET


class FSDPTrainer(Trainer):
    """Trainer that saves a single consolidated checkpoint under FSDP.

    FSDP shards the weights across GPUs; this gathers them to rank-0 on CPU and
    writes one ordinary `save_pretrained` checkpoint that inference can load.
    """
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        wrapped = getattr(self, "model_wrapped", self.model)
        if not isinstance(wrapped, FSDP):
            return super().save_model(output_dir, _internal_call=_internal_call)
        policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT, policy):
            cpu_state_dict = wrapped.state_dict()
        if self.args.should_save:
            self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)


def select_semantic_targets(hidden_states, labels, layout):
    """Gather sparse predictor states for Mimi codebook 0 and speech EOS.

    ``hidden_states[:, t]`` predicts ``labels[:, t + 1]``. Returning only the
    semantic positions lets fused SimPO avoid a prohibitively large
    ``sequence_length x vocabulary`` logits allocation.
    """
    shifted_labels = labels[:, 1:]
    predictor_states = hidden_states[:, :-1]
    relative = shifted_labels - (layout.base + AUDIO_OFFSET)
    is_semantic = (relative >= 0) & (relative < layout.codebook_size)
    selected = is_semantic | (shifted_labels == layout.eos_speech)
    if not torch.all(selected.any(dim=-1)):
        raise ValueError("every SimPO sequence needs a selected audio target")
    state_rows = []
    label_rows = []
    for row in range(predictor_states.shape[0]):
        positions = torch.nonzero(selected[row], as_tuple=False).squeeze(-1)
        state_rows.append(predictor_states[row, positions])
        label_rows.append(shifted_labels[row, positions])
    return (
        torch.nn.utils.rnn.pad_sequence(state_rows, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(
            label_rows, batch_first=True, padding_value=-100
        ),
    )


class SimPOTrainer(FSDPTrainer):
    """Fused, reference-free preference training on WER-ranked audio pairs."""

    def __init__(
        self,
        *args,
        layout,
        beta=2.0,
        gamma=1.0,
        sft_weight=0.1,
        wer_gap_scale=1.0,
        compiled=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.layout = layout
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.sft_weight = float(sft_weight)
        self.wer_gap_scale = float(wer_gap_scale)
        self.liger_simpo = LigerFusedLinearSimPOLoss(
            beta=self.beta,
            gamma=self.gamma,
            alpha=self.sft_weight,
            compute_nll_loss=self.sft_weight > 0.0,
            compiled=bool(compiled),
        )
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        labels = inputs["labels"]
        wer_gap = inputs.get("wer_gap")
        model_inputs = {
            key: value for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        model_inputs["use_cache"] = False
        outputs = model.base_model(**model_inputs)
        hidden_states, semantic_labels = select_semantic_targets(
            outputs.last_hidden_state,
            labels,
            self.layout,
        )
        lm_head = model.get_output_embeddings()
        loss, metrics = self.liger_simpo(
            lm_head.weight,
            hidden_states,
            semantic_labels,
            lm_head.bias,
        )
        if wer_gap is not None:
            step_weight = (
                1.0 + self.wer_gap_scale * wer_gap.clamp(0.0, 1.0).mean()
            )
            loss = loss * step_weight.to(loss.dtype)
        if return_outputs:
            return loss, {"simpo_metrics": metrics}
        return loss


class RatioTrainer(FSDPTrainer):
    """FSDPTrainer for the decaying text/speech pretraining recipe.

    Feeds the live training step into GradualRatioDataset, uses the
    interleave-preserving sampler, and logs separate text/speech losses.
    """
    def __init__(self, *args, initial_ratio, final_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.text_step = 0
        self.audio_step = 0
        self.total_steps = self._total_steps()
        if hasattr(self.train_dataset, "total_steps"):
            self.train_dataset.total_steps = self.total_steps

    def _total_steps(self):
        per_epoch = len(self.train_dataset) // (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps * self.args.world_size)
        return int(per_epoch * self.args.num_train_epochs)

    def current_ratio(self):
        if not self.total_steps:
            return self.initial_ratio
        progress = min(self.state.global_step / self.total_steps, 1.0)
        ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return max(int(round(ratio)), self.final_ratio)

    def get_train_dataloader(self):
        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank())
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler, collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0, pin_memory=self.args.dataloader_pin_memory)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if hasattr(self.train_dataset, "set_current_step"):
            self.train_dataset.set_current_step(self.state.global_step)
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if not (self.is_world_process_zero() and "loss" in logs):
            return
        ratio = self.current_ratio()
        wandb.log({"current_ratio": ratio, "global_step": self.state.global_step})
        # Within each (ratio + 1)-step cycle the first `ratio` steps are text.
        if self.state.global_step % (ratio + 1) < ratio:
            wandb.log({"text_loss": logs["loss"], "text_step": self.text_step})
            self.text_step += 1
        else:
            wandb.log({"audio_loss": logs["loss"], "audio_step": self.audio_step})
            self.audio_step += 1
