document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // current state
     cstate: {
      time: null,
      messages: [],
    },

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "Hello! Can you tell me about the benefits of exercise?",
          },
          {
            role: "assistant",
            content: "Certainly! Exercise has numerous benefits for both physical and mental health. Here are some key advantages:\n\n1. Improved cardiovascular health\n2. Increased muscle strength and endurance\n3. Better weight management\n4. Reduced risk of chronic diseases\n5. Enhanced mental health and mood\n6. Improved sleep quality\n7. Increased energy levels\n8. Better cognitive function\n\nRegular exercise can significantly improve your overall quality of life. It's recommended to aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity aerobic activity per week, along with muscle-strengthening activities at least twice a week.",
          },
          {
            role: "user",
            content: "That's great information! How can I start a simple exercise routine?",
          },
          {
            role: "assistant",
            content: "Starting a simple exercise routine is a great way to incorporate physical activity into your daily life. Here's a beginner-friendly approach:\n\n1. Start with walking: Aim for a 10-15 minute walk every day, gradually increasing the duration and pace.\n\n2. Add bodyweight exercises: Include simple exercises like push-ups, squats, and lunges. Start with 5-10 repetitions of each.\n\n3. Incorporate stretching: Spend 5-10 minutes stretching major muscle groups after your walk or exercises.\n\n4. Be consistent: Try to exercise at the same time each day to build a habit.\n\n5. Gradually increase intensity: As you get comfortable, increase the duration, add more exercises, or try jogging instead of walking.\n\n6. Stay hydrated and listen to your body: Drink plenty of water and rest when needed.\n\nRemember, the key is to start slow and gradually build up. Consult with a doctor before starting any new exercise program, especially if you have any health concerns.",
          },
          {
            role: "user",
            content: "Thank you! I'll start with a daily walk and bodyweight exercises.",
          },
          {
            role: "assistant",
            content: "You're welcome! That's a great way to begin. Remember to listen to your body and enjoy the process. If you have any questions or need further guidance, feel free to ask. I'm here to help!",
          },
        ],
      },
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
        {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
        {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
        {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
        {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
        {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
        {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      },
      {
        time: Date.now(),
        messages: [
          {
            role: "user",
            content: "What are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options? hat are some healthy breakfast options?",
          }
        ]
      }
    ],

    home: 0,
    generating: false,
    showHistoryModal: false,
    endpoint: `${window.location.origin}/v1`,
    

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    async handleSend() {
      const el = document.getElementById("input-form");
      const value = el.value.trim();
      if (!value) return;

      if (this.generating) return;
      this.generating = true;
      if (this.home === 0) this.home = 1;

      // ensure that going back in history will go back to home
      window.history.pushState({}, "", "/");

      // add message to list
      this.cstate.messages.push({ role: "user", content: value });

      // clear textarea
      el.value = "";
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";

      // reset performance tracking
      const prefill_start = Date.now();
      let start_time = 0;
      let tokens = 0;
      this.tokens_per_second = 0;

      // start receiving server sent events
      let gottenFirstChunk = false;
      for await (const chunk of this.openaiChatCompletion(
        this.cstate.messages
      )) {
        if (!gottenFirstChunk) {
          this.cstate.messages.push({ role: "assistant", content: "" });
          gottenFirstChunk = true;
        }

        // add chunk to the last message
        this.cstate.messages[this.cstate.messages.length - 1].content += chunk;

        // calculate performance tracking
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

      // update the state in histories or add it if it doesn't exist
      const index = this.histories.findIndex((cstate) => {
        return cstate.time === this.cstate.time;
      });
      this.cstate.time = Date.now();
      if (index !== -1) {
        // update the time
        this.histories[index] = this.cstate;
      } else {
        this.histories.push(this.cstate);
      }
      // update in local storage
      localStorage.setItem("histories", JSON.stringify(this.histories));

      this.generating = false;
    },

    async handleEnter(event) {
      // if shift is not pressed
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
      // stream response
      const response = await fetch(`${this.endpoint}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: messages,
          stream: true,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch");
      }

      const reader = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new EventSourceParserStream())
        .getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (value.type === "event") {
          const json = JSON.parse(value.data);
          if (json.choices) {
            const choice = json.choices[0];
            if (choice.finish_reason === "stop") {
              break;
            }
            yield choice.delta.content;
          }
        }
      }
    },
  }));
});

const { markedHighlight } = globalThis.markedHighlight;
marked.use(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code, lang, _info) {
      const language = hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
  })
);

// **** eventsource-parser ****
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
  let isFirstChunk;
  let buffer;
  let startingPosition;
  let startingFieldLength;
  let eventId;
  let eventName;
  let data;
  reset();
  return {
    feed,
    reset,
  };
  function reset() {
    isFirstChunk = true;
    buffer = "";
    startingPosition = 0;
    startingFieldLength = -1;
    eventId = void 0;
    eventName = void 0;
    data = "";
  }
  function feed(chunk) {
    buffer = buffer ? buffer + chunk : chunk;
    if (isFirstChunk && hasBom(buffer)) {
      buffer = buffer.slice(BOM.length);
    }
    isFirstChunk = false;
    const length = buffer.length;
    let position = 0;
    let discardTrailingNewline = false;
    while (position < length) {
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") {
          ++position;
        }
        discardTrailingNewline = false;
      }
      let lineLength = -1;
      let fieldLength = startingFieldLength;
      let character;
      for (
        let index = startingPosition;
        lineLength < 0 && index < length;
        ++index
      ) {
        character = buffer[index];
        if (character === ":" && fieldLength < 0) {
          fieldLength = index - position;
        } else if (character === "\r") {
          discardTrailingNewline = true;
          lineLength = index - position;
        } else if (character === "\n") {
          lineLength = index - position;
        }
      }
      if (lineLength < 0) {
        startingPosition = length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }
      parseEventStreamLine(buffer, position, fieldLength, lineLength);
      position += lineLength + 1;
    }
    if (position === length) {
      buffer = "";
    } else if (position > 0) {
      buffer = buffer.slice(position);
    }
  }
  function parseEventStreamLine(lineBuffer, index, fieldLength, lineLength) {
    if (lineLength === 0) {
      if (data.length > 0) {
        onParse({
          type: "event",
          id: eventId,
          event: eventName || void 0,
          data: data.slice(0, -1),
          // remove trailing newline
        });

        data = "";
        eventId = void 0;
      }
      eventName = void 0;
      return;
    }
    const noValue = fieldLength < 0;
    const field = lineBuffer.slice(
      index,
      index + (noValue ? lineLength : fieldLength)
    );
    let step = 0;
    if (noValue) {
      step = lineLength;
    } else if (lineBuffer[index + fieldLength + 1] === " ") {
      step = fieldLength + 2;
    } else {
      step = fieldLength + 1;
    }
    const position = index + step;
    const valueLength = lineLength - step;
    const value = lineBuffer.slice(position, position + valueLength).toString();
    if (field === "data") {
      data += value ? "".concat(value, "\n") : "\n";
    } else if (field === "event") {
      eventName = value;
    } else if (field === "id" && !value.includes("\0")) {
      eventId = value;
    } else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!Number.isNaN(retry)) {
        onParse({
          type: "reconnect-interval",
          value: retry,
        });
      }
    }
  }
}
const BOM = [239, 187, 191];
function hasBom(buffer) {
  return BOM.every((charCode, index) => buffer.charCodeAt(index) === charCode);
}
