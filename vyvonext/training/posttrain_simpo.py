"""WER-ranked SimPO post-training for the English voice-cloning model."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, TrainingArguments

from vyvonext.core.data import make_preference_collator
from vyvonext.core.tokens import TokenLayout
from vyvonext.core.trainer import SimPOTrainer


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "posttrain_simpo.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def parquet_files(path):
    path = os.fspath(path)
    if any(character in path for character in "*?[") or path.endswith(".parquet"):
        return path
    return os.path.join(path, "*.parquet")


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"])
    layout = TokenLayout.from_tokenizer(tokenizer)
    raw = load_dataset(
        "parquet",
        data_files=parquet_files(data_cfg["dataset"]),
        split="train",
    )
    raw = raw.filter(
        lambda row: (
            float(row["wer_gap"])
            >= float(data_cfg.get("minimum_wer_gap", 0.05))
            and len(row["chosen_input_ids"]) <= int(data_cfg["max_total_tokens"])
            and len(row["rejected_input_ids"]) <= int(data_cfg["max_total_tokens"])
        ),
        desc="validating WER preference pairs",
    ).shuffle(seed=train_cfg.get("seed", 42))
    if len(raw) < 2:
        raise ValueError("no usable WER preference pairs remain after filtering")

    validation_fraction = float(data_cfg.get("validation_fraction", 0.05))
    if validation_fraction > 0.0:
        split = raw.train_test_split(
            test_size=validation_fraction, seed=train_cfg.get("seed", 42)
        )
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = raw, None

    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_cfg["checkpoint"],
        attn_implementation=model_cfg.get(
            "attn_implementation", "kernels-community/flash-attn2"
        ),
        torch_dtype=torch.bfloat16,
    )
    expected_vocab = layout.base + layout.num_added_tokens + 1
    if model.config.vocab_size != expected_vocab:
        raise ValueError(
            f"checkpoint vocab={model.config.vocab_size}, expected {expected_vocab}; "
            "use the same base tokenizer and Mimi layout as pretraining"
        )
    model.config.use_cache = False
    evaluation = train_cfg.get("eval_strategy", "epoch") if eval_dataset else "no"
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        run_name=train_cfg.get("run_name"),
        num_train_epochs=train_cfg.get("epochs", 0.25),
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg.get("learning_rate", 5.0e-7),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
        warmup_steps=train_cfg.get("warmup_steps", 2),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        logging_steps=train_cfg.get("logging_steps", 1),
        eval_strategy=evaluation,
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=True,
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to=train_cfg.get("report_to", "wandb"),
        remove_unused_columns=False,
        prediction_loss_only=True,
        average_tokens_across_devices=False,
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=True,
    )
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_preference_collator(data_cfg["pad_token"]),
        layout=layout,
        beta=loss_cfg.get("beta", 2.0),
        gamma=loss_cfg.get("gamma", 1.0),
        sft_weight=loss_cfg.get("sft_weight", 0.1),
        wer_gap_scale=loss_cfg.get("wer_gap_scale", 1.0),
        compiled=loss_cfg.get("compiled", True),
    )
    print(
        f"WER-SimPO: {len(train_dataset)} train / "
        f"{len(eval_dataset) if eval_dataset is not None else 0} eval pairs"
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.config.use_cache = True
    trainer.save_model(train_cfg["output_dir"])


if __name__ == "__main__":
    main()
