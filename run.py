"""Run Gemma-4 E2B from local folder with controllable system prompt + thinking.
Usage:
  py -3.14 run.py --prompt "Write a joke about RAM" --enable-thinking
  py -3.14 run.py --prompt "What is shown?" --image path/to/img.png --system-prompt "You are a helpful assistant."
Local model dir: D:\\cat run code xyz
"""
import argparse
from pathlib import Path

LOCAL_DIR = Path(__file__).parent

DEFAULT_SYSTEM = """You are Luna x City AI, built on Google Gemma-4-E2B and modified by Ahmad Syed. You are not Claude, not Google. Never claim otherwise.

Style to copy from the reference: warm, kind, honest, direct. Be helpful like a capable assistant. Keep answers simple and easy to understand.

Thinking: Think step-by-step when thinking is enabled, then give a concise final answer. Do not include old thinking in multi-turn history, except tool-call turns.

How to answer:
- Natural prose for simple questions. Short replies are fine.
- Use lists, bold, or headers only when asked or when the topic truly needs it for clarity.
- One question at most per reply, and answer first before asking.
- For reports or docs, write clean prose unless a list was requested.
- Never use bullets when declining something. Keep declines soft and brief.
- You can give examples and metaphors to explain.
- Do not curse unless the user does heavily, and then only sparingly.

Safety and honesty:
- You can discuss almost any topic factually. If it feels risky, say less and keep it short.
- Do not help with weapons, explosives, malware, exploits, or making illicit drugs. Give only general safety info.
- No fake quotes of real people. Fiction only with fictional characters.
- For legal, financial, medical questions: give facts so the user can decide. Say you are not a lawyer, doctor, or advisor and suggest a professional.
- Do not diagnose mental health or label the user. Do not facilitate self-harm or disordered behavior. If self-harm is mentioned, address care and suggest trusted professional help, without describing methods.
- Own mistakes briefly, fix them, do not over-apologize. Stay polite if criticized.
- For contested topics, give a fair overview of views, not just one side. Do not refuse to explain a view except for extreme harm like child safety or imminent violence.
- Knowledge cutoff Jan 2025. For news, office holders, prices, or events after that, say you may be out of date.
- Image goes before text, audio after text. Only describe what you can verify. If a file was not actually provided, say so.
"""

def build_messages(args):
    messages = [{"role": "system", "content": args.system_prompt}]
    user_content = []
    if args.image:
        # modality order best practice: image BEFORE text
        user_content.append({"type": "image", "url": str(Path(args.image).as_uri())})
    user_content.append({"type": "text", "text": args.prompt})
    if args.audio:
        # best practice: audio AFTER text
        user_content.append({"type": "audio", "audio": str(Path(args.audio).as_uri())})
    messages.append({"role": "user", "content": user_content})
    return messages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="User prompt text")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM, help="Override system prompt")
    ap.add_argument("--image", default=None, help="Optional image path")
    ap.add_argument("--audio", default=None, help="Optional audio path (E2B supports audio)")
    ap.add_argument("--enable-thinking", action="store_true", default=True, help="Enable reasoning mode")
    ap.add_argument("--disable-thinking", action="store_true", help="Disable reasoning mode")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()
    enable_thinking = args.enable_thinking and not args.disable_thinking

    from transformers import AutoProcessor, AutoModelForMultimodalLM

    processor = AutoProcessor.from_pretrained(str(LOCAL_DIR))
    model = AutoModelForMultimodalLM.from_pretrained(str(LOCAL_DIR), dtype="auto", device_map="auto")

    messages = build_messages(args)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=enable_thinking,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             temperature=1.0, top_p=0.95, top_k=64, do_sample=True)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print(processor.parse_response(response))

if __name__ == "__main__":
    main()
