import http.server, socketserver, os
os.chdir('/tmp/browser-poc')

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # COOP/COEP -> crossOriginIsolated => SharedArrayBuffer => threads WASM.
        # 'credentialless' garde les ressources CDN (pas besoin de CORP).
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def log_message(self, *a):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('0.0.0.0', 8095), H) as s:
    s.serve_forever()
