import argparse
import os
import uuid
from dataclasses import dataclass, field

import gradio as gr
import torch
from gradio.themes.utils import colors, fonts, sizes

from chat_llm.engine import GenerateEngine
from chat_llm.model.llm import build_model_meta
from chat_llm.tokenizer import get_tokenizer
from chat_llm.utils.common import autodetect_device_type

# -----------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="Gradio chat app")
parser.add_argument("--source", type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument("--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("--step", type=int, default=None, help="Step to load")
parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for generation")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
parser.add_argument(
    "--device-type",
    type=str,
    default="",
    choices=["", "cuda", "cpu", "mps"],
    help="Device type for evaluation: cuda|cpu|mps. empty => autodetect",
)
parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio host")
parser.add_argument("--port", type=int, default=7860, help="Gradio port")
args = parser.parse_args()


# -----------------------------------------------------------------------------
# Environment / device
TOKENIZER_DIR = "./weight"
MODEL_DIR = "./weight"

device = autodetect_device_type() if args.device_type == "" else args.device_type


# -----------------------------------------------------------------------------
# Model config
@dataclass
class ModelConfig:
    depth: int = 26
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 4096
    window_pattern: str = "SSSL"
    vocab_size: int = 32000
    steps: int = 7000


config = ModelConfig()

print(os.getcwd())

# -----------------------------------------------------------------------------
# Load tokenizer
tokenizer = get_tokenizer(TOKENIZER_DIR)
config.vocab_size = tokenizer.get_vocab_size()


# -----------------------------------------------------------------------------
# Build model
model = build_model_meta(
    depth=config.depth,
    aspect_ratio=config.aspect_ratio,
    head_dim=config.head_dim,
    vocab_size=config.vocab_size,
    max_seq_len=config.max_seq_len,
    window_pattern=config.window_pattern,
)
model.to_empty(device=device)
model.init_weights()


def resolve_step(model_dir: str, step: int | None, fallback_step: int) -> int:
    if step is not None:
        return step

    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pt")]
    if model_files:
        return max(int(f[len("model_") : -len(".pt")]) for f in model_files)

    return fallback_step


step = resolve_step(MODEL_DIR, args.step, config.steps)
model_path = os.path.join(MODEL_DIR, f"model_{step:06d}.pt")
assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"

model_data = torch.load(model_path, map_location=device)
if model_data is not None:
    model.load_state_dict(model_data)

model.eval()
engine = GenerateEngine(model, tokenizer)


# -----------------------------------------------------------------------------
# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")


# -----------------------------------------------------------------------------
# Session state
@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    conversation_tokens: list[int] = field(default_factory=lambda: [bos])


def trim_context(tokens: list[int], max_seq_len: int, reserve_new_tokens: int) -> list[int]:
    max_prompt_len = max_seq_len - reserve_new_tokens
    if len(tokens) <= max_prompt_len:
        return tokens

    # keep BOS if present, then preserve the most recent suffix
    if tokens and tokens[0] == bos:
        keep = [bos]
        suffix_len = max_prompt_len - 1
        if suffix_len <= 0:
            return [bos]
        return keep + tokens[-suffix_len:]
    return tokens[-max_prompt_len:]


def build_prompt_tokens(conversation_tokens: list[int], user_text: str) -> list[int]:
    tokens = list(conversation_tokens)
    tokens.append(user_start)
    tokens.extend(tokenizer.encode(user_text))
    tokens.append(user_end)
    tokens.append(assistant_start)
    tokens = trim_context(tokens, config.max_seq_len, args.max_new_tokens)
    return tokens


def append_assistant_response(
    conversation_tokens: list[int],
    user_text: str,
    response_tokens: list[int],
) -> list[int]:
    updated = list(conversation_tokens)
    updated.append(user_start)
    updated.extend(tokenizer.encode(user_text))
    updated.append(user_end)
    updated.append(assistant_start)

    response_tokens = list(response_tokens)
    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)

    updated.extend(response_tokens)
    updated = trim_context(updated, config.max_seq_len, 0)
    return updated


def new_session_state() -> SessionState:
    return SessionState()


# -----------------------------------------------------------------------------
# Chat functions
def on_new_session():
    state = new_session_state()
    return [], state, f"Session: {state.session_id}"


def on_clear_chat(state: SessionState | None):
    _ = state
    state = new_session_state()
    return [], state, f"Session: {state.session_id}"


def stream_chat(user_message, chat_history, state: SessionState | None):
    if state is None:
        state = new_session_state()

    if chat_history is None:
        chat_history = []

    user_message = (user_message or "").strip()
    if not user_message:
        yield chat_history, state, ""
        return

    prompt_tokens = build_prompt_tokens(state.conversation_tokens, user_message)

    chat_history = list(chat_history)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})

    response_tokens: list[int] = []

    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    with torch.no_grad():
        for token_column, _token_masks in engine.generate(prompt_tokens, **generate_kwargs):
            token = int(token_column[0])

            if token == assistant_end:
                response_tokens.append(token)
                break

            response_tokens.append(token)
            partial_text = tokenizer.decode(response_tokens)
            chat_history[-1]["content"] = partial_text
            yield chat_history, state, ""

    state.conversation_tokens = append_assistant_response(
        state.conversation_tokens,
        user_message,
        response_tokens,
    )

    yield chat_history, state, ""


# -----------------------------------------------------------------------------
# UI
title = "NanoChat"
description = "Scientific chat interface with streaming, persistent session context, and lightweight experiment-style controls."

scientific_theme = gr.themes.Soft(
    primary_hue=colors.slate,
    secondary_hue=colors.blue,
    neutral_hue=colors.gray,
    font=[fonts.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=sizes.radius_lg,
)

custom_css = """
:root {
  --radius-xl: 16px;
}

.gradio-container {
  max-width: 1120px !important;
}

body, .gradio-container {
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
  letter-spacing: 0.01em;
}

#topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

#app-title {
  font-size: 30px;
  font-weight: 700;
  line-height: 1.15;
  letter-spacing: -0.02em;
}

#session-badge {
  font-size: 12px;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  opacity: 0.9;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--border-color-primary);
  background: rgba(255, 255, 255, 0.04);
}

#chatbot {
  min-height: 68vh;
}

#chatbot .message, .message {
  border-radius: 14px !important;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
}

textarea, input, button {
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;
}

code, pre, .monospace {
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
}

footer {
  visibility: hidden;
}
"""

with gr.Blocks(title=title) as demo:
    init_state = new_session_state()
    state = gr.State(init_state)

    with gr.Row(elem_id="topbar"):
        gr.Markdown("## NanoChat · Scientific Interface", elem_id="app-title")
        session_text = gr.Markdown(f"Session: {init_state.session_id}", elem_id="session-badge")

    chatbot = gr.Chatbot(
        value=[],
        elem_id="chatbot",
        avatar_images=(None, None),
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Message NanoChat...",
            lines=1,
            scale=8,
            container=True,
            autofocus=True,
        )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        new_btn = gr.Button("New Session")
        clear_btn = gr.Button("Clear")

    msg.submit(
        fn=stream_chat,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, msg],
    )

    send_btn.click(
        fn=stream_chat,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, msg],
    )

    new_btn.click(
        fn=on_new_session,
        inputs=[],
        outputs=[chatbot, state, session_text],
    )

    clear_btn.click(
        fn=on_clear_chat,
        inputs=[state],
        outputs=[chatbot, state, session_text],
    )

    gr.Markdown(
        f"""
**Run Configuration**  
Model source: `{args.source}`  
Model tag: `{args.model_tag}`  
Checkpoint step: `{step}`  
Device: `{device}`
"""
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        css=custom_css,
        theme=scientific_theme,
    )
