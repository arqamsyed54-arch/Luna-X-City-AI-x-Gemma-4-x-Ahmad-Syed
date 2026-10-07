from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        html = b"""<html><body style="font-family:sans-serif;max-width:700px;margin:40px auto">
<h1>Luna x City AI - test server OK</h1>
<p>If you see this at <b>http://127.0.0.1:7860</b>, your browser + firewall are fine.</p>
<p>Full model chat needs <b>torch+transformers+gradio</b> + 10GB load. Keep this window open, then open the URL.</p>
</body></html>"""
        self.send_response(200); self.send_header("Content-Type","text/html"); self.send_header("Content-Length",str(len(html))); self.end_headers()
        self.wfile.write(html)
    def log_message(self,*a): print(a)
HTTPServer(("127.0.0.1",7860),H).serve_forever()
