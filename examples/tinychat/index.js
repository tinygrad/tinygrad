document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // Current state
    cstate: {
      time: null,
      messages: [],
    },
    // Historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [
      {
        time: Date.now(),
        messages: [
          { role: "user", content: "Hello! I'm a chatbot trained on the OpenAI GPT-3 model. How can I help you today?" },
          {role: "assistant", content: "What is the capital of France?"},
        ]
      }
    ],
    home: 0,
    generating: false,
    showHistoryModal: false,
    endpoint: `${window.location.origin}/v1`,
    // Performance tracking
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

      // Ensure going back in history returns to home
      window.history.pushState({}, "", "/");

      // Add user message
      this.cstate.messages.push({ role: "user", content: value });

      // Clear textarea
      el.value = "";
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";

      // Reset performance tracking
      const prefill_start = Date.now();
      let start_time = 0;
      let tokens = 0;
      this.tokens_per_second = 0;

      // Receive server-sent events
      let gottenFirstChunk = false;
      for await (const chunk of this.openaiChatCompletion(this.cstate.messages)) {
        if (!gottenFirstChunk) {
          this.cstate.messages.push({ role: "assistant", content: "" });
          gottenFirstChunk = true;
        }

        // Append chunk to the last message
        this.cstate.messages[this.cstate.messages.length - 1].content += chunk;

        // Update performance tracking
        tokens += 1;
        this.total_tokens += 1;
        if (start_time === 0) {
          start_time = Date.now();
          this.time_till_first = start_time - prefill_start;
        } else {
          const diff = Date.now() - start_time;
          if (diff > 0) {
            this.tokens_per_second = tokens / (diff / 1000);
          }
        }
      }

      // Update histories
      const index = this.histories.findIndex((cstate) => cstate.time === this.cstate.time);
      this.cstate.time = Date.now();
      if (index !== -1) {
        this.histories[index] = this.cstate;
      } else {
        this.histories.push(this.cstate);
      }
      localStorage.setItem("histories", JSON.stringify(this.histories));

      this.generating = false;
    },

    async handleEnter(event) {
      if (!event.shiftKey) {
        event.preventDefault();
        await this.handleSend();
      }
    },

    updateTotalTokens(messages) {
      fetch(`${this.endpoint}/chat/token/encode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
      })
        .then((response) => response.json())
        .then((data) => {
          this.total_tokens = data.length;
        })
        .catch(console.error);
    },

    async *openaiChatCompletion(messages) {
      // Stream response
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
          if (json.choices) {
            const choice = json.choices[0];
            if (choice.finish_reason === "stop") break;
            yield choice.delta.content;
          }
        }
      }
    },
  }));
});

// Configure marked for syntax highlighting
marked.use(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code, lang) {
      const language = hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
  })
);

// EventSourceParserStream implementation
class EventSourceParserStream extends TransformStream {
  constructor() {
    let parser;
    super({
      start(controller) {
        parser = createParser((event) => {
          if (event.type === "event") {
            controller.enqueue(event);
          }
        });
      },
      transform(chunk) {
        parser.feed(chunk);
      },
    });
  }
}

function createParser(onParse) {
  let isFirstChunk = true;
  let buffer = "";
  let startingPosition = 0;
  let startingFieldLength = -1;
  let eventId, eventName, data = "";

  return {
    feed,
    reset: () => {
      isFirstChunk = true;
      buffer = "";
      startingPosition = 0;
      startingFieldLength = -1;
      eventId = undefined;
      eventName = undefined;
      data = "";
    },
  };

  function feed(chunk) {
    buffer += chunk;
    if (isFirstChunk && buffer.startsWith("\uFEFF")) {
      buffer = buffer.slice(1);
    }
    isFirstChunk = false;
    parseBuffer();
  }

  function parseBuffer() {
    let position = 0;
    let discardTrailingNewline = false;
    while (position < buffer.length) {
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") position++;
        discardTrailingNewline = false;
      }
      let lineLength = -1;
      let fieldLength = startingFieldLength;
      for (let index = startingPosition; lineLength < 0 && index < buffer.length; ++index) {
        const char = buffer[index];
        if (char === ":" && fieldLength < 0) {
          fieldLength = index - position;
        } else if (char === "\r") {
          discardTrailingNewline = true;
          lineLength = index - position;
        } else if (char === "\n") {
          lineLength = index - position;
        }
      }
      if (lineLength < 0) {
        startingPosition = buffer.length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }
      parseLine(buffer.slice(position, position + lineLength), fieldLength);
      position += lineLength + 1;
    }
    if (position === buffer.length) {
      buffer = "";
    } else if (position > 0) {
      buffer = buffer.slice(position);
    }
  }

  function parseLine(line, fieldLength) {
    if (line.length === 0) {
      if (data.length > 0) {
        onParse({ type: "event", id: eventId, event: eventName || undefined, data: data.slice(0, -1) });
        data = "";
        eventId = undefined;
      }
      eventName = undefined;
      return;
    }
    const noValue = fieldLength < 0;
    const field = line.slice(0, noValue ? line.length : fieldLength);
    let value = "";
    if (!noValue) {
      const step = line[fieldLength + 1] === " " ? fieldLength + 2 : fieldLength + 1;
      value = line.slice(step);
    }
    if (field === "data") {
      data += value ? `${value}\n` : "\n";
    } else if (field === "event") {
      eventName = value;
    } else if (field === "id" && !value.includes("\0")) {
      eventId = value;
    } else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!isNaN(retry)) {
        onParse({ type: "reconnect-interval", value: retry });
      }
    }
  }
}
