"""Luna x City AI - browser chat UI (stdlib only, no torch needed for UI)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HTML = b"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Luna x City AI</title>
<style>body{font-family:sans-serif;max-width:760px;margin:0 auto;padding:16px}#c{border:1px solid #ccc;border-radius:12px;height:55vh;overflow:auto;padding:12px;margin:12px 0}.u{text-align:right;margin:8px}.u span{background:#e6f0ff;padding:8px 12px;border-radius:12px;display:inline-block}.a span{background:#f2f2f2;padding:8px 12px;border-radius:12px;display:inline-block}.row{display:flex;gap:8px}input{flex:1;padding:10px;border-radius:10px;border:1px solid #ccc}button{padding:10px 16px;border-radius:10px}</style>
</head><body>
<h2>Luna x City AI</h2><p style="color:#555">Built on Google Gemma-4-E2B, modified by Ahmad Syed. UI preview - full 10GB weights load separately.</p>
<div id="c"></div>
<div class="row"><input id="m" placeholder="Ask Luna..." onkeydown="if(event.key==='Enter')send()"><button onclick="send()">Send</button></div>
<script>let h=[];async function send(){let m=document.getElementById('m');let t=m.value.trim();if(!t)return;m.value='';let c=document.getElementById('c');c.innerHTML+='<div class=u><span>'+t+'</span></div>';c.scrollTop=c.scrollHeight;let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t,history:h})});let j=await r.json();h.push([t,j.reply]);c.innerHTML+='<div class=a><span>'+j.reply+'</span></div>';c.scrollTop=c.scrollHeight;}</script>
</body></html>"""

REAL_MODEL = None
REAL_PROC = None
LOAD_ERROR = "real weights disabled for stability on 16GB CPU (10GB BF16 leaves ~2GB free, generation OOMs). Set LUNA_REAL=1 to try."
import os as _os
if _os.environ.get("LUNA_REAL") == "1":
    try:
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        _d = str(Path(__file__).parent)
        print("Trying real weights load (10GB, slow on CPU)...", flush=True)
        REAL_PROC = AutoProcessor.from_pretrained(_d)
        REAL_MODEL = AutoModelForMultimodalLM.from_pretrained(_d, dtype="auto", device_map="auto")
        print("Real model loaded on", REAL_MODEL.device, flush=True)
    except Exception as e:
        LOAD_ERROR = str(e)[:500]
        print("Real weights not loaded, demo mode:", LOAD_ERROR, flush=True)
else:
    print("Demo mode (stable). To try real 10GB load: $env:LUNA_REAL='1'; py server_chat.py", flush=True)

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","text/html"); self.send_header("Content-Length",str(len(HTML))); self.end_headers()
        self.wfile.write(HTML)
    def do_POST(self):
        n = int(self.headers.get("Content-Length",0))
        data = json.loads(self.rfile.read(n) or b"{}")
        msg = data.get("message","")
        if REAL_MODEL is not None:
            try:
                sys = "You are Luna x City AI, built on Google Gemma-4-E2B and modified by Ahmad Syed. Be warm, simple, honest."
                messages=[{"role":"system","content":sys},{"role":"user","content":msg}]
                try:
                    inputs=REAL_PROC.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors="pt",add_generation_prompt=True,enable_thinking=False).to(REAL_MODEL.device)
                except Exception:
                    prompt=f"System: {sys}\nUser: {msg}\nAssistant:"
                    tok=getattr(REAL_PROC,"tokenizer",None) or REAL_PROC
                    enc=tok(prompt,return_tensors="pt")
                    import torch as _t
                    _ids=enc["input_ids"].to(REAL_MODEL.device)
                    inputs={"input_ids":_ids}
                    if "attention_mask" in enc:
                        inputs["attention_mask"]=enc["attention_mask"].to(REAL_MODEL.device)
                il=inputs["input_ids"].shape[-1]
                out=REAL_MODEL.generate(**inputs,max_new_tokens=64,temperature=1.0,top_p=0.95,top_k=64,do_sample=True)
                try:
                    reply=str(REAL_PROC.decode(out[0][il:],skip_special_tokens=True))[:4000]
                except Exception:
                    reply=str(REAL_PROC.tokenizer.decode(out[0][il:],skip_special_tokens=True))[:4000]
                if not reply.strip():
                    reply="(base model gave empty output - this base E2B is not instruction-tuned; use E2B-it for chat)"
            except Exception as e:
                reply="Model error: "+str(e)[:500]
        else:
            reply="(Demo UI - real 10GB Gemma weights not loaded yet: "+(LOAD_ERROR or "torch/transformers missing")+") Your message was: "+msg+" | Install requirements-web.txt and restart server_chat.py for real answers."
        b=json.dumps({"reply":reply}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers()
        self.wfile.write(b)
    def log_message(self,*a): print(a)

print("Serving chat at http://127.0.0.1:7860", flush=True)
HTTPServer(("127.0.0.1",7860),H).serve_forever()
