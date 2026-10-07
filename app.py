# Luna x City AI - browser chat (local server, open in browser)
# Install: py -3.14 -m pip install -r requirements-web.txt
# Run: py -3.14 app.py
# Then open: http://127.0.0.1:7860
from pathlib import Path
import gradio as gr
from transformers import AutoProcessor, AutoModelForMultimodalLM

LOCAL_DIR = str(Path(__file__).parent)

DEFAULT_SYSTEM = open(Path(__file__).parent / "run.py", encoding="utf-8").read().split('DEFAULT_SYSTEM = """')[1].split('"""')[0]

print("Loading processor + model from", LOCAL_DIR)
processor = AutoProcessor.from_pretrained(LOCAL_DIR)
model = AutoModelForMultimodalLM.from_pretrained(LOCAL_DIR, dtype="auto", device_map="auto")
print("Loaded on", model.device)

def chat_fn(message, history, system_prompt, enable_thinking, max_tokens):
    messages = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM}]
    for h_user, h_asst in history:
        messages.append({"role": "user", "content": h_user})
        messages.append({"role": "assistant", "content": h_asst})
    messages.append({"role": "user", "content": message})
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=bool(enable_thinking),
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=int(max_tokens),
                         temperature=1.0, top_p=0.95, top_k=64, do_sample=True)
    resp = processor.decode(out[0][input_len:], skip_special_tokens=False)
    return str(processor.parse_response(resp))

with gr.Blocks(title="Luna x City AI (Gemma-4-E2B based, by Ahmad Syed)") as demo:
    gr.Markdown("# Luna x City AI\nBuilt on Google Gemma-4-E2B, modified by Ahmad Syed. Runs locally, Apache-2.0 base.")
    sys = gr.Textbox(value="You are Luna x City AI, built on Google Gemma-4-E2B and modified by Ahmad Syed. Be warm, simple, honest.",
                     label="System prompt", lines=3)
    think = gr.Checkbox(value=True, label="Enable thinking")
    maxtok = gr.Slider(128, 2048, value=1024, step=64, label="Max new tokens")
    gr.ChatInterface(fn=chat_fn, additional_inputs=[sys, think, maxtok],
                     title="Chat in your browser")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
