import os
import json
import torch
import transformers
import evaluate
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from config import TrainingConfig
from data_loader import DataCollatorSpeechSeq2SeqWithPadding
from utils import clean_text


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [clean_text(s) for s in pred_str]
    label_str = [clean_text(s) for s in label_str]

    pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not pairs:
        return {"wer": 0.0}

    final_preds, final_labels = zip(*pairs)
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(
        predictions=list(final_preds), references=list(final_labels)
    )

    print("\n" + "=" * 50)
    print(f"PRED: {final_preds[0]}")
    print(f"GT:   {final_labels[0]}")
    print("=" * 50 + "\n")

    return {"wer": wer}


class WhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        if has_labels:
            labels = inputs.pop("labels")
        else:
            labels = None

        generated_tokens = model.generate(
            input_features=inputs["input_features"],
            max_new_tokens=440,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            return_dict_in_generate=True,
            output_scores=True,
        )

        if isinstance(generated_tokens, dict):
            generated_tokens = generated_tokens.sequences

        return (None, generated_tokens, labels)


def build_training_args(config: TrainingConfig, output_dir: str = None) -> Seq2SeqTrainingArguments:
    if output_dir is None:
        output_dir = config.output_dir

    # Parse JSON string fields
    try:
        label_names = json.loads(config.label_names)
    except (json.JSONDecodeError, TypeError):
        label_names = ["labels"]

    try:
        grad_ckpt_kwargs = json.loads(config.gradient_checkpointing_kwargs)
    except (json.JSONDecodeError, TypeError):
        grad_ckpt_kwargs = {"use_reentrant": False}

    report_to = config.report_to
    if report_to == "wandb" and not config.wandb_api_key:
        report_to = "none"

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams,
        eval_accumulation_steps=config.eval_accumulation_steps,
        logging_steps=config.logging_steps,
        remove_unused_columns=config.remove_unused_columns,
        label_names=label_names,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        report_to=report_to,
        generation_config=None,
        prediction_loss_only=config.prediction_loss_only,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=grad_ckpt_kwargs,
    )


def create_trainer(
    model,
    processor,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: TrainingConfig,
    training_args: Seq2SeqTrainingArguments = None,
):
    if training_args is None:
        training_args = build_training_args(config)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, max_label_length=config.max_label_length
    )

    trainer = WhisperTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
    )

    return trainer


def run_training(trainer: WhisperTrainer, config: TrainingConfig, log_queue=None, stop_event=None):
    def log_callback(msg):
        if log_queue:
            log_queue.put(msg)
        print(msg)

    resume_from_checkpoint = None
    if config.resume_from_checkpoint:
        if os.path.exists(config.output_dir):
            checkpoints = [
                d for d in os.listdir(config.output_dir)
                if d.startswith("checkpoint")
            ]
            if checkpoints:
                resume_from_checkpoint = True
                log_callback(f"Resuming from checkpoint in {config.output_dir}")

    log_callback("Starting training...")

    class StopCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if stop_event and stop_event.is_set():
                control.should_training_stop = True
            return control

    if stop_event:
        trainer.add_callback(StopCallback())

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = f"{config.output_dir}/final_model"
    trainer.save_model(final_dir)
    if hasattr(trainer, "processor"):
        trainer.processor.save_pretrained(final_dir)
    log_callback(f"Training finished. Model saved in {final_dir}")
