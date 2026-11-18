import http.server, functools
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin'); self.send_header('Cross-Origin-Embedder-Policy','require-corp'); super().end_headers()
    def log_message(self,*a): pass
http.server.ThreadingHTTPServer(('127.0.0.1',8080),functools.partial(H,directory='.')).serve_forever()
