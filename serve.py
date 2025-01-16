from http.server import SimpleHTTPRequestHandler, HTTPServer
import time

hostName = "0.0.0.0"
serverPort = 8000

class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, post-check=0, pre-check=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "Thu, 01 Dec 1994 16:00:00 GMT")
        super().end_headers()

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), NoCacheHTTPRequestHandler)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")