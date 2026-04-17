import json, requests, threading, time, unittest

class FakeTokenizer:
  def __init__(self):
    self.preset = 'qwen2'
    self.bos_id = None
    self.eos_id = 0
    self.eot_id = None
  def role(self, role): return []
  def encode(self, text): return [ord(c) for c in text]
  def decode(self, ids): return ''.join(chr(i) for i in ids if i != 0)
  def stream_decoder(self):
    return lambda tid=None: '' if tid is None else ('' if tid == 0 else chr(tid))
  def end_turn(self): return []
  def prefix(self): return []
  def is_end(self, tid): return tid == 0

class FakeModel:
  def __init__(self):
    self.output = ''
    self.calls = []
  def get_start_pos(self, ids): return 0
  def generate(self, ids, **kwargs):
    self.calls.append(kwargs)
    return iter([*(ord(c) for c in self.output), 0])

class TestLLMXGrammarServer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from tinygrad.llm.cli import LLMServer
    cls.model = FakeModel()
    cls.server = LLMServer(('127.0.0.1', 0), cls.model, 'test-model', FakeTokenizer())
    cls.port = cls.server.server_address[1]
    cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.thread.start()
    time.sleep(0.1)

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def url(self, path='/v1/chat/completions'):
    return f'http://127.0.0.1:{self.port}{path}'

  def test_non_streaming_json_schema_response(self):
    self.model.output = '{"city":"Paris","temp_c":21}'
    resp = requests.post(self.url(), json={
      'model': 'test-model',
      'messages': [{'role': 'user', 'content': 'weather'}],
      'response_format': {
        'type': 'json_schema',
        'json_schema': {
          'name': 'weather',
          'schema': {
            'type': 'object',
            'properties': {'city': {'type': 'string'}, 'temp_c': {'type': 'integer'}},
            'required': ['city', 'temp_c'],
            'additionalProperties': False,
          }
        }
      }
    }, timeout=10)
    data = resp.json()
    self.assertEqual(json.loads(data['choices'][0]['message']['content'])['city'], 'Paris')
    self.assertIn('constraint', self.model.calls[-1])

  def test_non_streaming_tool_call_response(self):
    self.model.output = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    resp = requests.post(self.url(), json={
      'model': 'test-model',
      'messages': [{'role': 'user', 'content': 'weather'}],
      'tools': [{
        'type': 'function',
        'function': {
          'name': 'get_weather',
          'parameters': {
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'additionalProperties': False,
          }
        }
      }],
      'tool_choice': 'required'
    }, timeout=10)
    data = resp.json()
    msg = data['choices'][0]['message']
    self.assertEqual(data['choices'][0]['finish_reason'], 'tool_calls')
    self.assertEqual(msg['tool_calls'][0]['function']['name'], 'get_weather')
    self.assertEqual(json.loads(msg['tool_calls'][0]['function']['arguments'])['city'], 'Paris')

  def test_non_streaming_parallel_tool_calls_response(self):
    self.model.output = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n<tool_call>\n{"name": "get_time", "arguments": {"timezone": "Europe/Paris"}}\n</tool_call>'
    resp = requests.post(self.url(), json={
      'model': 'test-model',
      'messages': [{'role': 'user', 'content': 'weather and time'}],
      'tools': [
        {'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city'], 'additionalProperties': False}}},
        {'type': 'function', 'function': {'name': 'get_time', 'parameters': {'type': 'object', 'properties': {'timezone': {'type': 'string'}}, 'required': ['timezone'], 'additionalProperties': False}}}
      ],
      'tool_choice': 'required',
      'parallel_tool_calls': True
    }, timeout=10)
    data = resp.json()
    self.assertEqual(len(data['choices'][0]['message']['tool_calls']), 2)

  def test_streaming_tool_call_chunks(self):
    self.model.output = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    resp = requests.post(self.url(), json={
      'model': 'test-model',
      'messages': [{'role': 'user', 'content': 'weather'}],
      'tools': [{
        'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city'], 'additionalProperties': False}}
      }],
      'tool_choice': 'required',
      'stream': True
    }, stream=True, timeout=10)
    body = '\n'.join(line.decode() for line in resp.iter_lines() if line)
    self.assertIn('tool_calls', body)
    self.assertIn('get_weather', body)

if __name__ == '__main__':
  unittest.main()
