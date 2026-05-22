"""
Whisper Bilingual Training - CLI entry point
Run from cmd: python main.py [--config saved_config.json]
"""
import os
import sys
import argparse
import json
import wandb

from config import TrainingConfig
from model_setup import setup_model
from data_loader import load_all_datasets, load_all_test_datasets
from training import build_training_args, create_trainer, run_training
from utils import show_lang_stats_str, cleanup


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Bilingual Training CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to saved config JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume from checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not resume")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA visible devices (e.g. 0, 1)")
    parser.add_argument("--load_data_only", action="store_true", help="Only load and preview datasets, then exit")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()
        print("Using default config")

    # Override from CLI args
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.resume is not None:
        config.resume_from_checkpoint = args.resume
    if args.gpu is not None:
        config.cuda_visible_devices = args.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_SILENT"] = "true"

    if config.wandb_api_key:
        wandb.login(key=config.wandb_api_key)

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_all_datasets(config)
    test_dataset = load_all_test_datasets(config)
    print(show_lang_stats_str(train_dataset, "Train"))
    print(show_lang_stats_str(test_dataset, "Test"))

    if args.load_data_only:
        print("--load_data_only set. Exiting.")
        return

    # Setup model
    model, feature_extractor, tokenizer, processor = setup_model(config)

    # Build trainer
    training_args = build_training_args(config)
    trainer = create_trainer(model, processor, tokenizer, train_dataset, test_dataset, config, training_args)

    # Train
    run_training(trainer, config)

    cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
