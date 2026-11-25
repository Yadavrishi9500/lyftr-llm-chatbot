import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");
const MAX_MESSAGE_CHARS = 4000;

const SIMULATION_OPTIONS = [
  { value: "", label: "Normal" },
  { value: "key_1=429", label: "Simulate key_1=429" },
  { value: "key_1=500", label: "Simulate key_1=500" },
  { value: "*=429", label: "Simulate *=429" },
];

function newSessionId() {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === "function") return `s-${c.randomUUID().slice(0, 8)}`;
  return `s-${Math.random().toString(16).slice(2, 10).padEnd(8, "0")}`;
}

/** Turn any failed response into a { title, text } pair for the alert banner. */
function describeHttpError(status, body) {
  const detail = body && body.detail;
  let text;
  if (Array.isArray(detail)) {
    text = detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${d.msg}` : d.msg;
      })
      .join(" · ");
  } else if (typeof detail === "string") {
    text = detail;
  } else {
    text = `Request failed with status ${status}.`;
  }
  const titles = {
    422: "Invalid request (422)",
    503: "Service unavailable (503)",
    502: "Provider error (502)",
  };
  return { title: titles[status] || `Error (${status})`, text };
}

function ErrorBanner({ title, text }) {
  return (
    <div className="alert" role="alert" data-testid="error-banner">
      <span className="alert-icon" aria-hidden="true">!</span>
      <div>
        <div className="alert-title">{title}</div>
        <div className="alert-text">{text}</div>
      </div>
    </div>
  );
}

function AssistantMessage({ reply, metadata: m }) {
  const isCache = m.resolution_source === "LLM_CACHE";
  const failedOver = m.fallback_attempts > 0;
  return (
    <div className="row row-assistant">
      <div className="bubble bubble-assistant" data-testid="assistant-message">
        <div className="bubble-text">{reply}</div>
        <div className="meta">
          <span className="meta-item">
            <span className="meta-label">source</span>
            <span
              className={`pill ${isCache ? "pill-cache" : "pill-live"}`}
              data-testid="meta-source"
            >{m.resolution_source}</span>
          </span>
          <span className="meta-item">
            <span className="meta-label">model</span>
            <span className="meta-value" data-testid="meta-model">{m.model}</span>
          </span>
          <span className="meta-item">
            <span className="meta-label">latency</span>
            <span className="meta-value" data-testid="meta-latency">{`${m.latency_ms} ms`}</span>
          </span>
          <span className="meta-item">
            <span className="meta-label">in</span>
            <span className="meta-value" data-testid="meta-input-tokens">{`${m.input_tokens}`}</span>
          </span>
          <span className="meta-item">
            <span className="meta-label">out</span>
            <span className="meta-value" data-testid="meta-output-tokens">{`${m.output_tokens}`}</span>
          </span>
          {m.key_id && (
            <span
              className={`badge ${failedOver ? "badge-failover" : ""}`}
              data-testid="meta-key"
              title={failedOver ? "Served after automatic key failover" : "Key that served this reply"}
            >
              {failedOver ? `${m.key_id} · ${m.fallback_attempts} failover${m.fallback_attempts > 1 ? "s" : ""}` : m.key_id}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [items, setItems] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [loading, setLoading] = useState(false);

  const loadingRef = useRef(false);
  const nextId = useRef(1);
  const lastPromptRef = useRef("");
  const feedEndRef = useRef(null);
  const inputRef = useRef(null);

  const push = useCallback((item) => {
    setItems((prev) => [...prev, { id: nextId.current++, ...item }]);
  }, []);

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items, loading]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const send = useCallback(async () => {
    if (loadingRef.current || input.length === 0) return;
    const text = input;
    const prompt = systemPrompt.trim();

    setInput("");
    push({ kind: "user", content: text.trim() || "(whitespace only)" });
    loadingRef.current = true;
    setLoading(true);

    const body = { session_id: sessionId, message: text };
    // Send the prompt when set, and send "" once to clear a previously sent one.
    if (prompt || lastPromptRef.current) body.system_prompt = prompt;

    const headers = { "Content-Type": "application/json" };
    if (simulate) headers["X-Simulate-Failure"] = simulate;

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      let payload = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }
      if (!res.ok) {
        push({ kind: "error", ...describeHttpError(res.status, payload) });
      } else if (!payload || typeof payload.reply !== "string" || !payload.metadata) {
        push({ kind: "error", title: "Unexpected response", text: "The server returned a response this app could not read." });
      } else {
        lastPromptRef.current = prompt;
        push({ kind: "assistant", reply: payload.reply, metadata: payload.metadata });
      }
    } catch {
      push({
        kind: "error",
        title: "Network error",
        text: `Could not reach the API at ${API_BASE_URL}. Check that the backend is running and CORS allows this origin.`,
      });
    } finally {
      loadingRef.current = false;
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [input, systemPrompt, sessionId, simulate, push]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  };

  const startNewSession = () => {
    setSessionId(newSessionId());
    setItems([]);
    lastPromptRef.current = "";
    inputRef.current?.focus();
  };

  const overLimit = input.trim().length > MAX_MESSAGE_CHARS;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden="true">◆</span>
          <div>
            <h1>Lyftr Chat</h1>
            <div className="session-line">
              session <code>{sessionId}</code>
            </div>
          </div>
        </div>
        <div className="header-controls">
          <label className="sim">
            <span className="sr-only">Failure simulation</span>
            <select
              value={simulate}
              onChange={(e) => setSimulate(e.target.value)}
              className={simulate ? "sim-active" : ""}
              aria-label="Failure simulation"
              title="Sends an X-Simulate-Failure header (backend needs ENABLE_FAILURE_SIMULATION=true)"
            >
              {SIMULATION_OPTIONS.map((o) => (
                <option key={o.value || "normal"} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
          <button type="button" className="btn btn-ghost" onClick={startNewSession}>
            New Session
          </button>
        </div>
      </header>

      <main className="feed" aria-live="polite">
        {items.length === 0 && !loading && (
          <div className="empty">
            <div className="empty-title">Ask anything.</div>
            <div className="empty-sub">
              Replies show where they came from (live model or cache), tokens, latency and which key served them.
              Open a New Session and repeat a first message to see a cache hit.
            </div>
          </div>
        )}
        {items.map((item) => {
          if (item.kind === "user") {
            return (
              <div className="row row-user" key={item.id}>
                <div className="bubble bubble-user">{item.content}</div>
              </div>
            );
          }
          if (item.kind === "assistant") {
            return <AssistantMessage key={item.id} reply={item.reply} metadata={item.metadata} />;
          }
          return <ErrorBanner key={item.id} title={item.title} text={item.text} />;
        })}
        {loading && (
          <div className="row row-assistant" aria-label="Assistant is typing">
            <div className="bubble bubble-assistant typing"><span /><span /><span /></div>
          </div>
        )}
        <div ref={feedEndRef} />
      </main>

      <footer className="composer">
        <input
          type="text"
          className="system-input"
          data-testid="system-prompt-input"
          placeholder="System prompt (optional, applies to this session)"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          aria-label="System prompt"
          autoComplete="off"
        />
        <div className="composer-row">
          <textarea
            ref={inputRef}
            className="chat-input"
            data-testid="chat-input"
            placeholder="Message… (Enter to send, Shift+Enter for a new line)"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            aria-label="Message"
            autoFocus
          />
          <button
            type="button"
            className="btn btn-primary"
            data-testid="send-button"
            onClick={send}
            disabled={loading || input.length === 0}
          >
            Send
          </button>
        </div>
        <div className={`counter ${overLimit ? "counter-over" : ""}`}>
          {input.trim().length}/{MAX_MESSAGE_CHARS}
        </div>
      </footer>
    </div>
  );
}
