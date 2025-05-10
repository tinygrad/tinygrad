from http import HTTPStatus
from asyncio import StreamReader, StreamWriter

class HTTPServer:
  async def __call__(self, reader:StreamReader, writer:StreamWriter):
    while (req_hdr:=(await reader.readline()).decode().strip()):
      req_method, req_path, _ = req_hdr.split(' ')
      req_headers = {}
      while (hdr:=(await reader.readline()).decode().strip()):
        key, value = hdr.split(':', 1)
        req_headers[key.lower()] = value.strip()
      req_body = await reader.readexactly(int(req_headers.get("content-length", "0")))
      res_status, res_body = self.handle(req_method, req_path, req_headers, req_body)
      writer.write(f"HTTP/1.1 {res_status.value} {res_status.phrase}\r\nContent-Length: {len(res_body)}\r\n\r\n".encode() + res_body)
  def handle(self, method:str, path:str, headers:dict[str, str], body:bytes) -> tuple[HTTPStatus, bytes]: raise NotImplementedError('override')
