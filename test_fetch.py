from icecream import install
install()


import gzip, io, pathlib, unittest
from unittest.mock import patch
from tinygrad.helpers import fetch
from typing import Dict


class GzipResponse:
    def _gzip_content(self, content: str) -> bytes:
        stream = io.BytesIO()
        with gzip.GzipFile(fileobj=stream, mode='wb') as f: f.write(content.encode('utf-8'))
        return stream.getvalue()

    def __init__(self, content: str):
        self.gzipped_content: bytes = self._gzip_content(content)
        self.status: int = 200
        self.headers: Dict[str: str] = {'content-length': str(len(content))}
        self._position: int = 0

    def __enter__(self): return self

    def __exit__(self, exc_type, exc_val, exc_tb): pass

    def read(self, amount: int):
        self._position += amount
        return self.gzipped_content[self._position-amount:self._position]

class TestFetch(unittest.TestCase):

    @patch('urllib.request.urlopen')
    def _fetch_helper(self, mock_urlopen, content:str = "I love tinygrad!", gunzip:bool=True):
        # create mock response for urlopen
        mock_urlopen.return_value = GzipResponse(content)

        url = "http://example.com/data.gz"
        result = fetch(url, allow_caching=False, gunzip=gunzip)

        # assert statements
        mock_urlopen.assert_called_once()
        self.assertTrue(mock_urlopen.call_args[0][0] == url)
        self.assertTrue(isinstance(result, pathlib.Path) and result.is_file())
        with open(result, 'rb') as f:
            # file_content = gzip.decompress(f.read()).decode('utf-8')
            file_content = f.read().decode('utf-8')
        self.assertEqual(file_content, content)

    # def test_fetch(self):
    #     content = "I love tinygrad!"
    #     self._fetch_helper(content, gunzip:bool)



if __name__ == '__main__':
    unittest.main()
