import unittest, threading, time, json
from unittest.mock import Mock, patch
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.serve import StreamRouter, parse_tool_call

TEST_CONFIG = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                           norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)

class TestParseToolCall(unittest.TestCase):
  def test_argument_newlines(self):
    for value in ("", "text", " text ", "\n", "\nfirst\nsecond\n\n", "\r\nfirst\r\nsecond\r\n\r\n"):
      for newline in ("\n", "\r\n"):
        # GLM's values are exact (its template renders them as they are), the XML format puts them on their own lines
        self.assertEqual(parse_tool_call(f"write<arg_key>content</arg_key><arg_value>{newline}{value}{newline}</arg_value>"),
                         ("write", {"content":newline+value+newline}))
        self.assertEqual(parse_tool_call(f"<function=write><parameter=content>{newline}{value}{newline}</parameter></function>"),
                         ("write", {"content":value}))

  def test_glm_markup_in_values(self):
    # a value (code, a tool call in a string) can hold the format's own markup
    code = "parse_tool_call('write<arg_key>content</arg_key><arg_value>hi</arg_value>')"
    call = f"bash<arg_key>command</arg_key><arg_value>{code}</arg_value><arg_key>timeout</arg_key><arg_value>30</arg_value>"
    self.assertEqual(parse_tool_call(call), ("bash", {"command":code, "timeout":30}))
    self.assertEqual(parse_tool_call(f"bash<arg_key>command</arg_key><arg_value>{code}</arg_value>"), ("bash", {"command":code}))

  def test_special_token_markup(self):
    # tags that are special tokens decode to sentinels: the same text in a value is literal
    from tinygrad.llm.serve import markup_tags
    tags, sentinels = markup_tags({tag: 1000+i for i, tag in enumerate(("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
                                                                        "</arg_value>"))})
    self.assertEqual(len(sentinels), 6)
    code = "x = '</arg_value><arg_key>k</arg_key><arg_value>v</arg_value></tool_call>'"
    call = f"bash{tags['<arg_key>']}command{tags['</arg_key>']}{tags['<arg_value>']}{code}{tags['</arg_value>']}"
    self.assertEqual(parse_tool_call(call, tags), ("bash", {"command":code}))
    router, out = StreamRouter(tags=tags), []
    for piece in ("prose about `<tool_call>` tags ", tags["<tool_call>"], call, tags["</tool_call>"]): out += list(router.route(piece))
    out += list(router.route("", final=True))
    self.assertEqual("".join(d for f, d in out if f == "content"), "prose about `<tool_call>` tags ")
    self.assertTrue(router.buf.startswith(tags["<tool_call>"]))

  def test_json_arguments(self):
    for value in ('"text"', '42', 'true', 'null', '[1, "two"]', '{"nested":{"ok":true}}'):
      for call in (f"read<arg_key>value</arg_key><arg_value>{value}</arg_value>",
                   f"<function=read><parameter=value>{value}</parameter></function>"):
        self.assertEqual(parse_tool_call(call), ("read", {"value":json.loads(value)}))

  def test_glm_multiple_arguments(self):
    for separator in ("", "\n"):
      call = separator.join(("write", "<arg_key>path</arg_key>", "<arg_value>out.txt</arg_value>",
                             "<arg_key>content</arg_key>", "<arg_value>\nhello\n</arg_value>"))
      self.assertEqual(parse_tool_call(call), ("write", {"path":"out.txt", "content":"\nhello\n"}))

  def test_glm_no_arguments(self):
    self.assertEqual(parse_tool_call("tools.ping-v1"), ("tools.ping-v1", {}))

  def test_invalid_glm_call(self):
    for call in ("not a call", "read<arg_key>path</arg_key>", "read<arg_key>path</arg_key><arg_value>unfinished",
                 "read<arg_key>path</arg_key><arg_value>a</arg_value>trailing junk"):
      self.assertIsNone(parse_tool_call(call))

class TestLLMServer(unittest.TestCase):
  """Integration tests using the real OpenAI client."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="Hello")
    cls.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    cls.mock_tok.preset = "llama3"
    cls.mock_tok.bos_id = 1
    cls.mock_tok.eos_id = 999
    cls.mock_tok.eot_id = None
    cls.mock_tok.is_end = Mock(side_effect=lambda tid: tid in (999,))

    cls.mock_model = Mock()
    cls.mock_model.max_context = 4
    cls.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    cls.mock_model.get_start_pos = Mock(return_value=0)

    from tinygrad.llm.cli import FallbackTemplate
    from tinygrad.llm.serve import LLMServer

    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "test-model", cls.mock_tok, FallbackTemplate(cls.mock_tok))
    cls.port = cls.server.server_address[1]
    cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.server_thread.start()
    time.sleep(0.1)

    from openai import OpenAI
    cls.client = OpenAI(base_url=f"http://127.0.0.1:{cls.port}/v1", api_key="test")

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def test_chat_completion_stream(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
    self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

  def test_openai_response_structure(self):
    stream = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Test"}],
      stream=True
    )

    for chunk in stream:
      self.assertTrue(chunk.id.startswith("chatcmpl-"))
      self.assertEqual(chunk.object, "chat.completion.chunk")
      self.assertIsNotNone(chunk.choices)
      self.assertIsNotNone(chunk.created)
      self.assertIsInstance(chunk.created, int)
      self.assertEqual(chunk.model, "test-model")

  def test_stream_with_usage(self):
    def generate(ids, **kwargs):
      for token in (300, 301, 999):
        ids.append(token)
        yield token
    with patch.object(self.mock_model, "generate", side_effect=generate):
      chunks = list(self.client.chat.completions.create(
        model="test", messages=[{"role": "user", "content": "Hello"}], stream=True, stream_options={"include_usage": True}))
    last_chunk = chunks[-1]

    self.assertEqual(last_chunk.usage.prompt_tokens, 3)
    self.assertEqual(last_chunk.usage.completion_tokens, 2)
    self.assertEqual(last_chunk.usage.total_tokens, 5)

  def test_multi_turn_conversation(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
      ],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

  def test_text_parts_with_string_template(self):
    import jinja2
    # Qwen3's template concatenates content with strings, so a content array raises TypeError without normalization.
    template = jinja2.Template("{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{% endfor %}")
    with patch.object(self.server, "template", template), patch.object(self.mock_tok, "encode", side_effect=lambda text: [200, 201, 202]):
      response = self.client.chat.completions.create(model="test", messages=[
        {"role":"user", "content":[{"type":"text", "text":"Hello"}, {"type":"text", "text":" world"}]},
        {"role":"assistant", "content":None, "tool_calls":[
          {"id":"call_1", "type":"function", "function":{"name":"read", "arguments":"{}"}}]},
        {"role":"tool", "tool_call_id":"call_1", "content":"result"},
      ], stream=True)
      self.assertEqual(list(response)[-1].choices[0].finish_reason, "stop")
      self.mock_tok.encode.assert_called_with('<|im_start|>user\nHello world<|im_end|>\n'
                                              '<|im_start|>assistant\n<|im_end|>\n<|im_start|>tool\nresult<|im_end|>\n')

  def test_content_is_streamed(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    contents = []
    for chunk in stream:
      if chunk.choices and chunk.choices[0].delta.content:
        contents.append(chunk.choices[0].delta.content)

    self.assertGreater(len(contents), 0)

  def test_interrupted_stream_logs_tokens(self):
    # the threaded server checks cancellation between tokens, so gate the scheduler to synchronize:
    # generation pauses after the first token until close() has set req.cancelled
    gate, interrupted = threading.Event(), threading.Event()
    def slow_generate(ids, **kwargs):
      yield 300
      gate.wait(2)
      yield 301
    def log(s):
      if "total:" in s: interrupted.set()
    with patch.object(self.mock_model, "generate", side_effect=slow_generate), \
         patch("tinygrad.llm.serve.stderr_log", side_effect=log) as log_mock, \
         patch("tinygrad.llm.serve.colored", side_effect=lambda text, color: text) as color:
      stream = self.server.RequestHandlerClass.run_model(Mock(server=self.server), [200, 201, 202], "test")
      next(stream)
      next(stream)
      stream.close()
      gate.set()
      self.assertTrue(interrupted.wait(2))
    interrupt = log_mock.call_args.args[0]
    self.assertFalse(interrupt.startswith("\n"))
    self.assertTrue(interrupt.endswith("\n"))
    self.assertIn("gen:", interrupt)
    self.assertIn("out:    1", interrupt)
    self.assertTrue(any(args[0].startswith("total:") and args[1] == "red" for args, _ in color.call_args_list))

  def test_stream_disconnect_closes_source(self):
    from tinygrad.llm.serve import Handler
    source, handler = Mock(), Mock()
    source.__iter__ = Mock(return_value=iter([{}]))
    handler.wfile.write.side_effect = BrokenPipeError
    Handler.stream_json(handler, source)
    source.close.assert_called_once()

  def test_non_streaming(self):
    resp = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Hello"}],
      stream=False
    )

    self.assertTrue(resp.id.startswith("chatcmpl-"))
    self.assertEqual(resp.object, "chat.completion")
    self.assertEqual(resp.model, "test-model")
    self.assertIsNotNone(resp.created)
    self.assertEqual(len(resp.choices), 1)
    self.assertEqual(resp.choices[0].message.role, "assistant")
    self.assertIsNotNone(resp.choices[0].message.content)
    self.assertEqual(resp.choices[0].finish_reason, "stop")
    self.assertIsNotNone(resp.usage)
    self.assertIsNotNone(resp.usage.prompt_tokens)
    self.assertIsNotNone(resp.usage.completion_tokens)

  def test_context_length_error(self):
    from openai import BadRequestError
    self.mock_tok.encode.return_value = [200, 201, 202, 203]
    try:
      with self.assertRaises(BadRequestError) as err:
        self.client.chat.completions.create(model="test-model", messages=[{"role":"user", "content":"too long"}])
      self.assertEqual(err.exception.code, "context_length_exceeded")
    finally:
      self.mock_tok.encode.return_value = [200, 201, 202]

  def test_max_tokens_streaming(self):
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 302, 303, 999]))
    stream = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=True, max_tokens=2
    )
    chunks = list(stream)
    content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
    self.assertEqual(len(content_chunks), 2)
    self.assertEqual(chunks[-1].choices[0].finish_reason, "length")

  def test_max_tokens_non_streaming(self):
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 302, 303, 999]))
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=False, max_tokens=2
    )
    self.assertEqual(resp.choices[0].finish_reason, "length")
    self.assertEqual(resp.usage.completion_tokens, 2)

  def test_models_endpoint(self):
    import requests as req
    resp = req.get(f"http://127.0.0.1:{self.port}/v1/models")
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertEqual(data["object"], "list")
    self.assertEqual(len(data["data"]), 1)
    self.assertEqual(data["data"][0]["id"], "test-model")
    self.assertEqual(data["data"][0]["object"], "model")

  def test_concurrent_requests(self):
    # the server accepts and tokenizes a second request while the first is generating, but the inference loop runs them one at a time
    gate, generating, encoded, calls = threading.Event(), threading.Event(), threading.Event(), []
    def gated_generate(ids, **kwargs):
      calls.append(ids)
      generating.set()
      if len(calls) == 1: gate.wait(5)
      yield from (300, 999)
    def encode(text):
      if generating.is_set(): encoded.set()
      return [200, 201, 202]
    results: dict[str, str] = {}
    def chat(prompt):
      resp = self.client.chat.completions.create(model="test-model", messages=[{"role":"user", "content":prompt}])
      results[prompt] = resp.choices[0].message.content
    first, second = (threading.Thread(target=chat, args=(p,)) for p in ("first", "second"))
    with patch.object(self.mock_model, "generate", side_effect=gated_generate), patch.object(self.mock_tok, "encode", side_effect=encode):
      first.start()
      self.assertTrue(generating.wait(2))
      second.start()
      self.assertTrue(encoded.wait(2))
      self.assertEqual(len(calls), 1)
      gate.set()
      first.join(5)
      second.join(5)
    self.assertEqual(results, {"first":"Hello", "second":"Hello"})

  def test_disconnect_stops_generation(self):
    # after the client goes away the inference loop must stop at the req.cancelled check
    gate, done, generated_tokens = threading.Event(), threading.Event(), []
    def gated_generate(ids, **kwargs):
      for tid in (300, 301, 302, 999):
        if tid == 302: gate.wait(5)
        generated_tokens.append(tid)
        yield tid
    def log(s):
      if "total:" in s: done.set()
    with patch.object(self.mock_model, "generate", side_effect=gated_generate), patch("tinygrad.llm.serve.stderr_log", side_effect=log):
      stream = self.server.RequestHandlerClass.run_model(Mock(server=self.server), [200, 201, 202], "test")
      next(stream)
      next(stream)
      next(stream)
      stream.close()
      gate.set()
      self.assertTrue(done.wait(2))
    self.assertEqual(generated_tokens, [300, 301, 302])

  def test_model_generation_failure(self):
    from openai import APIError, InternalServerError
    def failing_generate(ids, **kwargs):
      yield 300
      raise RuntimeError("fake generate error")
    client = self.client.with_options(max_retries=0)  # the default client retries 500s, which would just rerun the failure
    with patch.object(self.mock_model, "generate", side_effect=failing_generate):
      # non-streaming: a 500 instead of a 200 with finish_reason "stop"
      with self.assertRaises(InternalServerError):
        client.chat.completions.create(model="test", messages=[{"role":"user","content":"Hello"}])
      # streaming: the client raises on the error chunk instead of returning the partial text
      with self.assertRaises(APIError):
        for _ in client.chat.completions.create(model="test", messages=[{"role":"user","content":"Hello"}], stream=True): pass

  def test_generate_runs_on_serving_thread(self):
    threads = []
    def generate(ids, **kwargs):
      threads.append(threading.current_thread())
      yield from (300, 999)
    with patch.object(self.mock_model, "generate", side_effect=generate):
      self.client.chat.completions.create(model="test-model", messages=[{"role":"user","content":"Hello"}])
    self.assertEqual(threads, [self.server_thread])

class TestLLMToolCalls(unittest.TestCase):
  """Tool calling through the OpenAI-compatible HTTP API."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="")
    cls.mock_tok.preset = "qwen2"
    cls.mock_tok.bos_id, cls.mock_tok.eos_id, cls.mock_tok.eot_id = None, 999, None
    cls.mock_tok.is_end = Mock(return_value=False)

    cls.mock_model = Mock()
    cls.mock_model.max_context = 4
    cls.mock_model.get_start_pos = Mock(return_value=0)

    from tinygrad.llm.serve import LLMServer
    import jinja2
    # .items() matches tool-aware templates and ensures OpenAI JSON argument strings are normalized before rendering the next turn.
    template = jinja2.Template("""{% for m in messages %}{{ m.content or '' }}{% for tc in m.tool_calls or [] %}
      {% for key, value in tc.function.arguments.items() %}{{ key }}={{ value }}{% endfor %}{% endfor %}{% endfor %}""")
    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "tool-model", cls.mock_tok, template)
    cls.port = cls.server.server_address[1]
    cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.server_thread.start()
    time.sleep(0.1)

    from openai import OpenAI
    cls.client = OpenAI(base_url=f"http://127.0.0.1:{cls.port}/v1", api_key="test")

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def set_output(self, text:str):
    pieces = dict(enumerate(text, 1))
    self.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: pieces[tid] if tid is not None else "")
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter(pieces))

  @staticmethod
  def tools():
    return [{"type":"function", "function":{"name":"read", "description":"Read a file",
      "parameters":{"type":"object", "properties":{"path":{"type":"string"}}, "required":["path"]}}}]

  def test_streaming_tool_call(self):
    self.set_output('before<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>')
    chunks = list(self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Read README.md"}],
                                                     tools=self.tools(), stream=True))
    self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks if c.choices), "before")
    calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0].function.name, "read")
    self.assertEqual(json.loads(calls[0].function.arguments), {"path":"README.md"})
    self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

  def test_multiple_xml_tool_calls(self):
    self.set_output("<tool_call><function=read><parameter=path>\"a\"</parameter></function></tool_call>"
                    "<tool_call><function=read><parameter=path>\"b\"</parameter></function></tool_call>")
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Read a and b"}],
                                                   tools=self.tools())
    self.assertEqual([json.loads(tc.function.arguments)["path"] for tc in response.choices[0].message.tool_calls], ["a", "b"])
    self.assertEqual(response.choices[0].finish_reason, "tool_calls")

  def test_streaming_glm_tool_calls(self):
    self.set_output("before<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value></tool_call>"
                    "<tool_call>read\n<arg_key>path</arg_key>\n<arg_value>\nb\n</arg_value>\n</tool_call>")
    chunks = list(self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Read files"}],
                                                     tools=self.tools(), stream=True))
    self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks if c.choices), "before")
    calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
    self.assertEqual([tc.function.name for tc in calls], ["read", "read"])
    # GLM's values are exact
    self.assertEqual([json.loads(tc.function.arguments) for tc in calls], [{"path":"a"}, {"path":"\nb\n"}])
    self.assertEqual([tc.index for tc in calls], [0, 1])
    self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

  def test_multiline_tool_argument_preserves_trailing_newline(self):
    self.set_output("<tool_call>\n<function=write>\n<parameter=content>\nfirst\nsecond\n\n</parameter>\n"
                    "<parameter=filePath>\nout.txt\n</parameter>\n</function>\n</tool_call>")
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Write out.txt"}], tools=self.tools())
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    self.assertEqual(args, {"content":"first\nsecond\n", "filePath":"out.txt"})

  def test_invalid_tool_call_becomes_content(self):
    self.set_output("<tool_call>not a call</tool_call>")
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Hello"}], tools=self.tools())
    self.assertEqual(response.choices[0].message.content, "<tool_call>not a call</tool_call>")
    self.assertIsNone(response.choices[0].message.tool_calls)
    self.assertEqual(response.choices[0].finish_reason, "stop")

  def test_tool_call_in_reasoning_is_not_executed(self):
    self.set_output('<think>draft <tool_call>{"name":"wrong","arguments":{}}</tool_call></think>answer')
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Hello"}], tools=self.tools())
    self.assertEqual(response.choices[0].message.content, "answer")
    self.assertIsNone(response.choices[0].message.tool_calls)
    self.assertEqual(response.choices[0].finish_reason, "stop")

  def test_tool_result_round_trip(self):
    self.set_output('<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>')
    first = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Read README.md"}], tools=self.tools())
    call = first.choices[0].message.tool_calls[0]
    self.set_output("done")
    second = self.client.chat.completions.create(model="tool-model", messages=[
      {"role":"user", "content":"Read README.md"},
      {"role":"assistant", "content":None, "tool_calls":[call.model_dump()]},
      {"role":"tool", "tool_call_id":call.id, "content":"file contents"},
    ], tools=self.tools())
    self.assertEqual(second.choices[0].message.content, "done")
    self.assertEqual(second.choices[0].finish_reason, "stop")

  # ** the Anthropic Messages API
  def anthropic(self, body:dict, path:str="/v1/messages") -> tuple[int, bytes]:
    import urllib.request, urllib.error
    req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    try:
      with urllib.request.urlopen(req) as resp: return resp.status, resp.read()
    except urllib.error.HTTPError as e: return e.code, e.read()

  @staticmethod
  def anthropic_tools(): return [{"name":"read", "description":"Read a file", "input_schema":{"type":"object", "properties":{"path":{"type":"string"}}}}]

  def test_anthropic_stream_tool_use(self):
    self.set_output('<think>plan</think>before<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>')
    status, data = self.anthropic({"model":"m", "max_tokens":100, "stream":True, "tools":self.anthropic_tools(),
                                   "messages":[{"role":"user", "content":"Read README.md"}]})
    self.assertEqual(status, 200)
    events = [json.loads(l[len("data: "):]) for l in data.decode().splitlines() if l.startswith("data: ")]
    names = [l[len("event: "):] for l in data.decode().splitlines() if l.startswith("event: ")]
    self.assertEqual(names, [e["type"] for e in events])
    self.assertEqual(events[0]["type"], "message_start")
    self.assertEqual(events[-1]["type"], "message_stop")
    blocks = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
    self.assertEqual(blocks, ["thinking", "text", "tool_use"])
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    self.assertEqual("".join(d.get("thinking", "") for d in deltas), "plan")
    self.assertEqual("".join(d.get("text", "") for d in deltas), "before")
    self.assertEqual([json.loads(d["partial_json"]) for d in deltas if d["type"] == "input_json_delta"], [{"path":"README.md"}])
    self.assertEqual(next(e for e in events if e["type"] == "message_delta")["delta"]["stop_reason"], "tool_use")

  def test_anthropic_round_trip(self):
    # the tool_use and tool_result blocks go back as the chat's tool calls and tool messages
    self.set_output("done")
    status, data = self.anthropic({"model":"m", "max_tokens":100, "system":[{"type":"text", "text":"be brief"}], "tools":self.anthropic_tools(),
      "messages":[{"role":"user", "content":"Read README.md"},
                  {"role":"assistant", "content":[{"type":"thinking", "thinking":"t", "signature":""},
                                                  {"type":"tool_use", "id":"toolu_1", "name":"read", "input":{"path":"README.md"}}]},
                  {"role":"user", "content":[{"type":"tool_result", "tool_use_id":"toolu_1", "content":"file contents"},
                                             {"type":"text", "text":"go on"}]}]})
    self.assertEqual(status, 200)
    reply = json.loads(data)
    self.assertEqual(reply["type"], "message")
    self.assertEqual(reply["role"], "assistant")
    self.assertEqual(reply["content"], [{"type":"text", "text":"done"}])
    self.assertEqual(reply["stop_reason"], "end_turn")
    self.assertEqual(reply["usage"]["output_tokens"], 4)

  def test_anthropic_messages_conversion(self):
    from tinygrad.llm.serve import anthropic_messages
    msgs, tools = anthropic_messages({"system":"sys", "tools":self.anthropic_tools() + [{"type":"web_search_20250305", "name":"web_search"}],
      "messages":[{"role":"user", "content":"q"},
                  {"role":"assistant", "content":[{"type":"text", "text":"a"}, {"type":"tool_use", "id":"t1", "name":"read", "input":{"path":"x"}}]},
                  {"role":"user", "content":[{"type":"tool_result", "tool_use_id":"t1", "content":[{"type":"text", "text":"r"}]}]}]})
    self.assertEqual(msgs, [{"role":"system", "content":"sys"}, {"role":"user", "content":"q"},
                            {"role":"assistant", "content":"a", "tool_calls":[{"id":"t1", "type":"function",
                                                                               "function":{"name":"read", "arguments":{"path":"x"}}}]},
                            {"role":"tool", "tool_call_id":"t1", "content":"r"}])
    self.assertEqual([t["function"]["name"] for t in tools or []], ["read"])

  def test_anthropic_count_tokens(self):
    status, data = self.anthropic({"model":"m", "messages":[{"role":"user", "content":"hi"}]}, "/v1/messages/count_tokens")
    self.assertEqual((status, json.loads(data)), (200, {"input_tokens":3}))

class TestTransformerGenerate(unittest.TestCase):
  def test_warmup(self):
    model, calls = Transformer(TEST_CONFIG), []
    def generate(tokens, **kwargs):
      calls.append(tokens)
      yield from (1, 2)
    with patch.object(model, "generate", generate): model.warmup()
    self.assertEqual(calls, [[0]])

  def test_template_starts_reasoning(self):
    router = StreamRouter(reasoning=True)
    self.assertEqual(list(router.route("reasoning</think>answer")),
                     [("reasoning_content", "reasoning"), ("content", "answer")])

if __name__ == '__main__':
  unittest.main()
