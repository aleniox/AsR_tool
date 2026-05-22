import os
import json
import threading
import queue
import io
import contextlib
import gradio as gr

from config import TrainingConfig
from utils import (
    show_lang_stats_str,
    cleanup,
)
from model_setup import setup_model
from data_loader import (
    load_all_datasets,
    load_all_test_datasets,
)
from training import build_training_args, create_trainer, run_training
from inference import transcribe

os.environ["WANDB_SILENT"] = "true"

CONFIG_SAVE_PATH = "saved_config.json"


class AppState:
    def __init__(self):
        self.config = TrainingConfig()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        self.train_dataset = None
        self.test_dataset = None
        self.trainer = None
        self.training_thread = None
        self.stop_event = None
        self.log_queue = None
        self.log_buffer = []
        self.training_running = False

    def cleanup_training(self):
        self.training_running = False
        self.training_thread = None
        self.stop_event = None
        self.log_queue = None
        cleanup()


state = AppState()


# ─── HELPERS ─────────────────────────────────────────────

def update_config_from_ui(
    model_name, output_dir, lr, batch_size, grad_accum, epochs,
    warmup, precision, augmentation, wandb_project, wandb_key,
    gpu_device, resume, max_test_samples,
    eval_batch_size, eval_steps, save_steps, logging_steps,
    save_total, max_label_len, gen_max_len, gen_beams,
    eval_strategy, save_strategy, predict_gen, remove_cols,
    label_names_str, load_best, metric_best, greater_better,
    report_to_val, pred_loss_only, grad_ckpt, grad_ckpt_kwargs,
    no_repeat, cond_prev, predict_ts, compress_ratio,
    logprob_thresh, no_speech_thresh,
):
    state.config.model_name_or_path = model_name
    state.config.output_dir = output_dir
    state.config.learning_rate = float(lr)
    state.config.per_device_train_batch_size = int(batch_size)
    state.config.gradient_accumulation_steps = int(grad_accum)
    state.config.num_train_epochs = int(epochs)
    state.config.warmup_steps = int(warmup)
    state.config.fp16 = precision == "fp16"
    state.config.bf16 = precision == "bf16"
    state.config.apply_augmentation = augmentation
    state.config.wandb_project = wandb_project
    state.config.wandb_api_key = wandb_key
    state.config.cuda_visible_devices = gpu_device
    state.config.resume_from_checkpoint = resume
    state.config.max_test_samples = int(max_test_samples)
    state.config.per_device_eval_batch_size = int(eval_batch_size)
    state.config.eval_steps = int(eval_steps)
    state.config.save_steps = int(save_steps)
    state.config.logging_steps = int(logging_steps)
    state.config.save_total_limit = int(save_total)
    state.config.max_label_length = int(max_label_len)
    state.config.generation_max_length = int(gen_max_len)
    state.config.generation_num_beams = int(gen_beams)
    state.config.eval_strategy = eval_strategy
    state.config.save_strategy = save_strategy
    state.config.predict_with_generate = predict_gen
    state.config.remove_unused_columns = remove_cols
    state.config.label_names = label_names_str
    state.config.load_best_model_at_end = load_best
    state.config.metric_for_best_model = metric_best
    state.config.greater_is_better = greater_better
    state.config.report_to = report_to_val
    state.config.prediction_loss_only = pred_loss_only
    state.config.gradient_checkpointing = grad_ckpt
    state.config.gradient_checkpointing_kwargs = grad_ckpt_kwargs
    state.config.no_repeat_ngram_size = int(no_repeat)
    state.config.condition_on_previous_text = cond_prev
    state.config.predict_timestamps = predict_ts
    state.config.compression_ratio_threshold = float(compress_ratio)
    state.config.logprob_threshold = float(logprob_thresh)
    state.config.no_speech_threshold = float(no_speech_thresh)
    state.config.save(CONFIG_SAVE_PATH)
    return f"Configuration saved to {CONFIG_SAVE_PATH}!"


def load_model_action():
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            state.model, state.feature_extractor, state.tokenizer, state.processor = setup_model(state.config)
        load_log = f.getvalue()
        return load_log.strip()
    except Exception as e:
        return f"Error loading model: {e}"


def load_data_action(
    local_train_str, local_test_str,
    online_train_str, online_test_str,
):
    f = io.StringIO()
    try:
        if local_train_str:
            paths = [p.strip() for p in local_train_str.split("\n") if p.strip()]
            state.config.local_train_datasets = paths
        if local_test_str:
            paths = [p.strip() for p in local_test_str.split("\n") if p.strip()]
            state.config.local_test_datasets = paths
        if online_train_str:
            try:
                state.config.online_train_datasets = json.loads(online_train_str)
            except json.JSONDecodeError:
                return "Invalid JSON for online train datasets"

        print("Loading and preprocessing train datasets...")
        with contextlib.redirect_stdout(f):
            state.train_dataset = load_all_datasets(state.config)
            state.test_dataset = load_all_test_datasets(state.config)
        load_log = f.getvalue()
        load_log = "\n".join(
            line for line in load_log.replace("\r", "\n").splitlines() if line.strip()
        )

        train_stats = show_lang_stats_str(state.train_dataset, "Train Dataset")
        test_stats = show_lang_stats_str(state.test_dataset, "Test Dataset")

        samples = []
        for i in range(min(5, len(state.train_dataset))):
            s = state.train_dataset[i]
            samples.append(f"#{i}: [{s['language']}] {s['sentence'][:100]}")

        return (
            f"{load_log}\n\n{train_stats}\n\n{test_stats}\n\n--- Preview ---\n" + "\n".join(samples)
        )
    except Exception as e:
        return f"Error loading data: {e}\n{f.getvalue()}"


# ─── TRAINING ─────────────────────────────────────────────

def start_training_action(model_name):
    if state.training_running:
        return "Training already running!", gr.update(interactive=False), gr.update(interactive=True)

    if state.train_dataset is None:
        return "Please load datasets first!", gr.update(interactive=False), gr.update(interactive=True)

    # Update model name from UI and always reload model to match config
    state.config.model_name_or_path = model_name
    from model_setup import setup_model
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            state.model, state.feature_extractor, state.tokenizer, state.processor = setup_model(state.config)
        load_log = f.getvalue()
    except Exception as e:
        return f"Error loading model '{model_name}': {e}", gr.update(interactive=False), gr.update(interactive=True)

    state.stop_event = threading.Event()
    state.log_queue = queue.Queue()

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = state.config.cuda_visible_devices
        if state.config.wandb_api_key:
            import wandb
            wandb.login(key=state.config.wandb_api_key)
        os.environ["WANDB_PROJECT"] = state.config.wandb_project

        training_args = build_training_args(state.config)
        state.trainer = create_trainer(
            state.model, state.processor, state.tokenizer,
            state.train_dataset, state.test_dataset,
            state.config, training_args,
        )
    except Exception as e:
        return f"Error setting up trainer: {e}", gr.update(interactive=False), gr.update(interactive=True)

    state.training_running = True
    state.training_thread = threading.Thread(
        target=run_training,
        args=(state.trainer, state.config, state.log_queue, state.stop_event),
        daemon=True,
    )
    state.training_thread.start()

    return (
        f"Training started! Model: {model_name}\n{load_log}",
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def stop_training_action():
    if state.stop_event:
        state.stop_event.set()
    state.training_running = False
    return (
        "Stop signal sent (will stop after current step).",
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def read_logs():
    if state.log_queue:
        while not state.log_queue.empty():
            try:
                state.log_buffer.append(state.log_queue.get_nowait())
            except queue.Empty:
                break
    return "\n".join(state.log_buffer)


def check_training_status():
    if state.training_thread and not state.training_thread.is_alive():
        if state.training_running:
            state.cleanup_training()
            return "Training completed!", gr.update(interactive=True), gr.update(interactive=False)
    return None, gr.update(), gr.update()


# ─── INFERENCE ─────────────────────────────────────────────

def run_inference(audio_input, model_path, language):
    if audio_input is None:
        return "Please upload or record audio."
    if not model_path:
        return "Please enter model path."
    try:
        result = transcribe(audio_input, model_path, language=language if language != "auto" else None)
        text = result["text"]
        return text
    except Exception as e:
        return f"Inference error: {e}"


def list_checkpoints(output_dir):
    if not os.path.exists(output_dir):
        return []
    items = os.listdir(output_dir)
    checkpoints = [d for d in items if d.startswith("checkpoint") or d == "final_model"]
    checkpoints.sort(reverse=True)
    return [os.path.join(output_dir, d) for d in checkpoints]


def refresh_checkpoints(output_dir):
    ckpts = list_checkpoints(output_dir)
    if not ckpts:
        return gr.update(choices=[], value=None)
    return gr.update(choices=ckpts, value=ckpts[0])


# ─── BUILD UI ─────────────────────────────────────────────

def build_app():
    # Load saved config if exists
    if os.path.exists(CONFIG_SAVE_PATH):
        try:
            state.config = TrainingConfig.load(CONFIG_SAVE_PATH)
        except Exception:
            pass

    with gr.Blocks(title="Whisper Bilingual Training UI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Whisper Bilingual Training & Inference UI")
        gr.Markdown("Train Whisper on Vietnamese + English bilingual datasets")

        with gr.Tabs():
            # ─── TAB 1: CONFIG ───────────────────────────
            with gr.TabItem("Config"):
                with gr.Group():
                    gr.Markdown("### Model & Output")
                    model_name = gr.Textbox(
                        label="Model Name or Path", value=state.config.model_name_or_path
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory", value=state.config.output_dir
                    )

                with gr.Group():
                    gr.Markdown("### Training Hyperparameters")
                    with gr.Row():
                        lr = gr.Number(label="Learning Rate", value=state.config.learning_rate, step=1e-6)
                        batch_size = gr.Number(label="Train Batch Size", value=state.config.per_device_train_batch_size, step=1, precision=0)
                        grad_accum = gr.Number(label="Gradient Accumulation Steps", value=state.config.gradient_accumulation_steps, step=1, precision=0)
                    with gr.Row():
                        epochs = gr.Number(label="Num Epochs", value=state.config.num_train_epochs, step=1, precision=0)
                        warmup = gr.Number(label="Warmup Steps", value=state.config.warmup_steps, step=10, precision=0)
                        eval_batch_size = gr.Number(label="Eval Batch Size", value=state.config.per_device_eval_batch_size, step=1, precision=0)
                    with gr.Row():
                        eval_steps = gr.Number(label="Eval Steps", value=state.config.eval_steps, step=100, precision=0)
                        save_steps = gr.Number(label="Save Steps", value=state.config.save_steps, step=100, precision=0)
                        logging_steps = gr.Number(label="Logging Steps", value=state.config.logging_steps, step=5, precision=0)
                    with gr.Row():
                        save_total = gr.Number(label="Save Total Limit", value=state.config.save_total_limit, step=1, precision=0)
                        max_label_len = gr.Number(label="Max Label Length", value=state.config.max_label_length, step=8, precision=0)
                        gen_max_len = gr.Number(label="Generation Max Length", value=state.config.generation_max_length, step=10, precision=0)
                    with gr.Row():
                        gen_beams = gr.Number(label="Num Beams", value=state.config.generation_num_beams, step=1, precision=0)
                        max_test = gr.Number(label="Max Test Samples", value=state.config.max_test_samples, step=100, precision=0)

                with gr.Group():
                    gr.Markdown("### Strategy & Reporting")
                    with gr.Row():
                        eval_strategy = gr.Dropdown(label="Eval Strategy", choices=["steps", "epoch", "no"], value=state.config.eval_strategy)
                        save_strategy = gr.Dropdown(label="Save Strategy", choices=["steps", "epoch", "no"], value=state.config.save_strategy)
                        report_to_val = gr.Dropdown(label="Report To", choices=["none", "wandb", "tensorboard", "all"], value=state.config.report_to)
                    with gr.Row():
                        metric_best = gr.Textbox(label="Metric for Best Model", value=state.config.metric_for_best_model)
                        label_names_str = gr.Textbox(label="Label Names (JSON)", value=state.config.label_names)
                        grad_ckpt_kwargs = gr.Textbox(label="Grad Checkpoint Kwargs (JSON)", value=state.config.gradient_checkpointing_kwargs)
                    with gr.Row():
                        predict_gen = gr.Checkbox(label="Predict with Generate", value=state.config.predict_with_generate)
                        remove_cols = gr.Checkbox(label="Remove Unused Columns", value=state.config.remove_unused_columns)
                        load_best = gr.Checkbox(label="Load Best Model at End", value=state.config.load_best_model_at_end)
                    with gr.Row():
                        greater_better = gr.Checkbox(label="Greater is Better", value=state.config.greater_is_better)
                        pred_loss_only = gr.Checkbox(label="Prediction Loss Only", value=state.config.prediction_loss_only)
                        grad_ckpt = gr.Checkbox(label="Gradient Checkpointing", value=state.config.gradient_checkpointing)

                with gr.Group():
                    gr.Markdown("### Generation Config (Anti-Hallucination)")
                    with gr.Row():
                        no_repeat = gr.Number(label="No Repeat Ngram Size", value=state.config.no_repeat_ngram_size, step=1, precision=0)
                        compress_ratio = gr.Number(label="Compression Ratio Threshold", value=state.config.compression_ratio_threshold, step=0.1)
                    with gr.Row():
                        logprob_thresh = gr.Number(label="Logprob Threshold", value=state.config.logprob_threshold, step=0.1)
                        no_speech_thresh = gr.Number(label="No Speech Threshold", value=state.config.no_speech_threshold, step=0.05)
                    with gr.Row():
                        cond_prev = gr.Checkbox(label="Condition on Previous Text", value=state.config.condition_on_previous_text)
                        predict_ts = gr.Checkbox(label="Predict Timestamps", value=state.config.predict_timestamps)

                with gr.Row():
                    precision = gr.Radio(
                        label="Precision",
                        choices=["fp16", "bf16", "none"],
                        value="fp16" if state.config.fp16 else ("bf16" if state.config.bf16 else "none"),
                    )
                    augmentation = gr.Checkbox(label="Apply Augmentation", value=state.config.apply_augmentation)
                    resume = gr.Checkbox(label="Resume from Checkpoint", value=state.config.resume_from_checkpoint)

                with gr.Group():
                    gr.Markdown("### WandB & GPU")
                    with gr.Row():
                        wandb_project = gr.Textbox(label="WandB Project", value=state.config.wandb_project)
                        wandb_key = gr.Textbox(label="WandB API Key", type="password", value=state.config.wandb_api_key)
                        gpu_device = gr.Textbox(label="CUDA Device", value=state.config.cuda_visible_devices)

                with gr.Row():
                    save_config_btn = gr.Button("Save Configuration", variant="primary")
                    load_model_btn = gr.Button("Load Model", variant="secondary")

                config_status = gr.Textbox(label="Status", interactive=False)

                save_config_btn.click(
                    fn=update_config_from_ui,
                    inputs=[model_name, output_dir, lr, batch_size, grad_accum,
                            epochs, warmup, precision, augmentation, wandb_project,
                            wandb_key, gpu_device, resume, max_test,
                            eval_batch_size, eval_steps, save_steps, logging_steps,
                            save_total, max_label_len, gen_max_len, gen_beams,
                            eval_strategy, save_strategy, predict_gen, remove_cols,
                            label_names_str, load_best, metric_best, greater_better,
                            report_to_val, pred_loss_only, grad_ckpt, grad_ckpt_kwargs,
                            no_repeat, cond_prev, predict_ts, compress_ratio,
                            logprob_thresh, no_speech_thresh],
                    outputs=config_status,
                )
                load_model_btn.click(
                    fn=load_model_action,
                    inputs=[],
                    outputs=config_status,
                )

            # ─── TAB 2: DATA ─────────────────────────────
            with gr.TabItem("Data"):
                gr.Markdown("### Dataset Configuration")
                gr.Markdown("Paths: one per line")

                with gr.Row():
                    with gr.Column():
                        local_train = gr.Textbox(
                            label="Local Train Datasets (paths)",
                            lines=6,
                            value="\n".join(TrainingConfig().local_train_datasets),
                        )
                        online_train = gr.Textbox(
                            label="Online Train Datasets (JSON list)",
                            lines=3,
                            placeholder='[{"path": "...", "split": "train", ...}]',
                        )
                    with gr.Column():
                        local_test = gr.Textbox(
                            label="Local Test Datasets (paths)",
                            lines=6,
                            value="\n".join(TrainingConfig().local_test_datasets),
                        )
                        online_test = gr.Textbox(
                            label="Online Test Datasets (JSON list)",
                            lines=3,
                            placeholder='[{"path": "...", "split": "test", ...}]',
                        )

                with gr.Row():
                    load_data_btn = gr.Button("Load & Preview Datasets", variant="primary")

                data_status = gr.Textbox(label="Dataset Info", lines=15, interactive=False)

                load_data_btn.click(
                    fn=load_data_action,
                    inputs=[local_train, local_test, online_train, online_test],
                    outputs=data_status,
                )

            # ─── TAB 3: TRAINING ─────────────────────────
            with gr.TabItem("Training"):
                with gr.Row():
                    start_btn = gr.Button("Start Training", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop Training", variant="stop", size="lg", interactive=False)

                train_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Training Logs")
                log_output = gr.Textbox(label="Logs", lines=20, interactive=False)

                refresh_logs_btn = gr.Button("Refresh Logs")

                # Training control
                start_btn.click(
                    fn=start_training_action,
                    inputs=[model_name],
                    outputs=[train_status, start_btn, stop_btn],
                )
                stop_btn.click(
                    fn=stop_training_action,
                    inputs=[],
                    outputs=[train_status, start_btn, stop_btn],
                )
                refresh_logs_btn.click(
                    fn=read_logs,
                    inputs=[],
                    outputs=log_output,
                )

                def auto_refresh():
                    status, start_upd, stop_upd = check_training_status()
                    logs = read_logs()
                    return status or "Idle", start_upd, stop_upd, logs

                training_timer = gr.Timer(3)
                training_timer.tick(
                    fn=auto_refresh,
                    inputs=[],
                    outputs=[train_status, start_btn, stop_btn, log_output],
                )

            # ─── TAB 4: INFERENCE ────────────────────────
            with gr.TabItem("Inference"):
                gr.Markdown("### Transcribe Audio")

                with gr.Row():
                    with gr.Column():
                        model_path_drop = gr.Dropdown(
                            label="Model Checkpoint",
                            choices=[],
                            interactive=True,
                        )
                        output_dir_box = gr.Textbox(
                            label="Output Directory",
                            value="weights/whisper-medium-bilingual-vi-en",
                        )
                        refresh_btn = gr.Button("Refresh Checkpoints")
                    with gr.Column():
                        lang_drop = gr.Dropdown(
                            label="Language",
                            choices=["auto", "vi", "en"],
                            value="auto",
                        )

                audio_input = gr.Audio(
                    label="Audio Input", type="filepath", sources=["upload", "microphone"]
                )

                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
                transcript_output = gr.Textbox(label="Transcript", lines=5)

                refresh_btn.click(
                    fn=refresh_checkpoints,
                    inputs=output_dir_box,
                    outputs=model_path_drop,
                )
                transcribe_btn.click(
                    fn=run_inference,
                    inputs=[audio_input, model_path_drop, lang_drop],
                    outputs=transcript_output,
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
