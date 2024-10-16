class EventSourceParserStream extends TransformStream {
  constructor() {
    let parser;
    super({
      start(controller) {
        parser = createParser((event) => {
          if (event.type === "event") controller.enqueue(event);
        });
      },
      transform(chunk) {
        parser.feed(chunk);
      },
    });
  }
}

function createParser(onParse) {
  let isFirstChunk = true,
    buffer = "",
    startingPosition = 0,
    startingFieldLength = -1,
    eventId,
    eventName,
    data = "";

  function reset() {
    isFirstChunk = true;
    buffer = "";
    startingPosition = 0;
    startingFieldLength = -1;
    eventId = undefined;
    eventName = undefined;
    data = "";
  }

  function feed(chunk) {
    buffer += chunk;
    if (isFirstChunk && hasBom(buffer)) buffer = buffer.slice(BOM.length);
    isFirstChunk = false;
    let position = 0,
      discardTrailingNewline = false;

    while (position < buffer.length) {
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") position++;
        discardTrailingNewline = false;
      }

      let lineLength = -1,
        fieldLength = startingFieldLength;

      for (let i = startingPosition; lineLength < 0 && i < buffer.length; ++i) {
        const char = buffer[i];
        if (char === ":" && fieldLength < 0) fieldLength = i - position;
        else if (char === "\r") {
          discardTrailingNewline = true;
          lineLength = i - position;
        } else if (char === "\n") lineLength = i - position;
      }

      if (lineLength < 0) {
        startingPosition = buffer.length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }

      parseEventStreamLine(buffer, position, fieldLength, lineLength);
      position += lineLength + 1;
    }
    buffer = buffer.slice(position);
  }

  function parseEventStreamLine(lineBuffer, index, fieldLength, lineLength) {
    if (lineLength === 0) {
      if (data) {
        onParse({ type: "event", id: eventId, event: eventName, data: data.slice(0, -1) });
        data = "";
        eventId = undefined;
      }
      eventName = undefined;
      return;
    }

    const noValue = fieldLength < 0;
    const field = lineBuffer.slice(index, index + (noValue ? lineLength : fieldLength));
    const step = noValue
      ? lineLength
      : lineBuffer[index + fieldLength + 1] === " "
      ? fieldLength + 2
      : fieldLength + 1;
    const value = lineBuffer.slice(index + step, index + lineLength).toString();

    if (field === "data") data += value ? `${value}\n` : "\n";
    else if (field === "event") eventName = value;
    else if (field === "id" && !value.includes("\0")) eventId = value;
    else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!isNaN(retry)) onParse({ type: "reconnect-interval", value: retry });
    }
  }

  return { feed, reset };
}

const BOM = [239, 187, 191];
const hasBom = (buffer) => BOM.every((charCode, index) => buffer.charCodeAt(index) === charCode);

document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    cstate: { time: null, messages: [] },
    histories: JSON.parse(localStorage.getItem("histories")) || [],
    home: 0,
    generating: false,
    showHistoryModal: false,
    endpoint: `${window.location.origin}/v1`,
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => state.time === cstate.time);
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    async handleSend() {
      const el = document.getElementById("input-form");
      const value = el.value.trim();
      if (!value || this.generating) return;
      this.generating = true;
      if (this.home === 0) this.home = 1;
      window.history.pushState({}, "", "/");
      this.cstate.messages.push({ role: "user", content: value });
      el.value = "";
      const prefill_start = Date.now();
      let start_time = 0,
        tokens = 0;

      for await (const chunk of this.openaiChatCompletion(this.cstate.messages)) {
        let lastMsg = this.cstate.messages[this.cstate.messages.length - 1];
        if (lastMsg.role !== "assistant") {
          lastMsg = { role: "assistant", content: "" };
          this.cstate.messages.push(lastMsg);
        }
        lastMsg.content += chunk;
        tokens++;
        this.total_tokens++;
        if (start_time === 0) {
          start_time = Date.now();
          this.time_till_first = start_time - prefill_start;
        } else {
          const diff = Date.now() - start_time;
          if (diff > 0) this.tokens_per_second = tokens / (diff / 1000);
        }
      }

      this.cstate.time = Date.now();
      this.histories.push(this.cstate);
      localStorage.setItem("histories", JSON.stringify(this.histories));
      this.generating = false;
    },

    handleEnter(event) {
      if (!event.shiftKey) {
        event.preventDefault();
        this.handleSend();
      }
    },

    updateTotalTokens(messages) {
      fetch(`${this.endpoint}/chat/token/encode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
      })
        .then((response) => response.json())
        .then((data) => (this.total_tokens = data.length))
        .catch(console.error);
    },

    async *openaiChatCompletion(messages) {
      const response = await fetch(`${this.endpoint}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, stream: true }),
      });
      if (!response.ok) throw new Error("Failed to fetch");

      const reader = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new EventSourceParserStream())
        .getReader();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value.type === "event") {
          const json = JSON.parse(value.data);
          const choice = json.choices?.[0];
          if (choice?.finish_reason === "stop") break;
          if (choice?.delta?.content) yield choice.delta.content;
        }
      }
    },
  }));
});

marked.use(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code, lang) {
      const language = hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
  })
);
