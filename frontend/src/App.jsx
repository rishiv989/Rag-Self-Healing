import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// ── Helpers ──────────────────────────────────────────────────────────────────

function strategyColor(strategy) {
  const map = {
    RAG: "#1dd4a9",
    CACHE: "#00d2ff",
    MEMORY: "#a78bfa",
    WEB_SEARCH: "#f59e0b",
    REPORT: "#60a5fa",
    REFUSE: "#f87171",
    CLARIFY: "#fb923c",
    MMR: "#34d399",
    REWRITE: "#818cf8",
  };
  return map[strategy] || "#a0a0b0";
}

function strategyIcon(strategy) {
  const map = {
    RAG: "🔍",
    CACHE: "⚡",
    MEMORY: "🧠",
    WEB_SEARCH: "🌐",
    REPORT: "📊",
    REFUSE: "🚫",
    CLARIFY: "❓",
    MMR: "🔀",
    REWRITE: "✏️",
  };
  return map[strategy] || "💬";
}

const GRAPH_NODES = [
  "analyze_query",
  "check_fast_memory",
  "retrieve_and_rerank",
  "self_heal",
  "web_search_fallback",
  "build_knowledge_graph",
  "generate_draft",
  "reflect",
];

// ── Simple Markdown Renderer (no library needed) ─────────────────────────────

function renderMarkdown(text) {
  if (!text) return [];
  const lines = text.split("\n");
  const elements = [];
  let i = 0;
  let keyCounter = 0;

  const key = () => `md-${keyCounter++}`;

  const inlineFormat = (str) => {
    // Code spans
    str = str.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
    // Bold
    str = str.replace(/\*\*([^*]+)\*\*/g, (_, c) => `<strong>${c}</strong>`);
    // Italic
    str = str.replace(/\*([^*]+)\*/g, (_, c) => `<em>${c}</em>`);
    return str;
  };

  while (i < lines.length) {
    const line = lines[i];

    // H1-H3
    if (line.startsWith("### ")) {
      elements.push(<h3 key={key()} className="md-h3" dangerouslySetInnerHTML={{ __html: inlineFormat(line.slice(4)) }} />);
    } else if (line.startsWith("## ")) {
      elements.push(<h2 key={key()} className="md-h2" dangerouslySetInnerHTML={{ __html: inlineFormat(line.slice(3)) }} />);
    } else if (line.startsWith("# ")) {
      elements.push(<h1 key={key()} className="md-h1" dangerouslySetInnerHTML={{ __html: inlineFormat(line.slice(2)) }} />);
    }
    // Horizontal rule
    else if (line.match(/^---+$/)) {
      elements.push(<hr key={key()} className="md-hr" />);
    }
    // Fenced code block
    else if (line.startsWith("```")) {
      const lang = line.slice(3).trim();
      const codeLines = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      elements.push(
        <div key={key()} className="md-code-block">
          {lang && <span className="md-code-lang">{lang}</span>}
          <pre><code>{codeLines.join("\n")}</code></pre>
        </div>
      );
    }
    // Bullet list
    else if (line.match(/^[-*] /)) {
      const items = [];
      while (i < lines.length && lines[i].match(/^[-*] /)) {
        items.push(<li key={key()} dangerouslySetInnerHTML={{ __html: inlineFormat(lines[i].slice(2)) }} />);
        i++;
      }
      elements.push(<ul key={key()} className="md-list">{items}</ul>);
      continue;
    }
    // Numbered list
    else if (line.match(/^\d+\. /)) {
      const items = [];
      while (i < lines.length && lines[i].match(/^\d+\. /)) {
        const content = lines[i].replace(/^\d+\. /, "");
        items.push(<li key={key()} dangerouslySetInnerHTML={{ __html: inlineFormat(content) }} />);
        i++;
      }
      elements.push(<ol key={key()} className="md-list">{items}</ol>);
      continue;
    }
    // Blockquote
    else if (line.startsWith("> ")) {
      elements.push(
        <blockquote key={key()} className="md-blockquote" dangerouslySetInnerHTML={{ __html: inlineFormat(line.slice(2)) }} />
      );
    }
    // Empty line = paragraph break (skip)
    else if (line.trim() === "") {
      // Add small spacer
    }
    // Normal paragraph text
    else {
      elements.push(
        <p key={key()} className="md-p" dangerouslySetInnerHTML={{ __html: inlineFormat(line) }} />
      );
    }
    i++;
  }
  return elements;
}

// ── Toast Component ──────────────────────────────────────────────────────────

function Toast({ message, type, onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, 4000);
    return () => clearTimeout(t);
  }, [onClose]);

  return (
    <div className={`toast toast-${type}`}>
      <span className="toast-icon">{type === "success" ? "✓" : type === "error" ? "✕" : "⟳"}</span>
      <span>{message}</span>
    </div>
  );
}

// ── Analytics Dashboard ──────────────────────────────────────────────────────

function AnalyticsDashboard() {
  const [data, setData] = useState(null);
  const [topQueries, setTopQueries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/analytics`).then((r) => r.json()),
      fetch(`${API_BASE}/analytics/top-queries?top_k=8`).then((r) => r.json()),
    ])
      .then(([analytics, queries]) => {
        setData(analytics);
        setTopQueries(queries.top_queries || []);
      })
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="analytics-loading">
        <div className="spinner" />
        <p>Loading analytics...</p>
      </div>
    );
  }

  if (!data || data.total_failures === 0) {
    return (
      <div className="analytics-empty">
        <div className="empty-icon">📊</div>
        <h3>No failures logged yet</h3>
        <p>The system hasn't encountered any retrieval failures. Keep asking questions!</p>
      </div>
    );
  }

  const maxStrategyCount = Math.max(...Object.values(data.strategy_counts));
  const maxReasonCount = Math.max(...Object.values(data.reason_counts));

  return (
    <div className="analytics-panel">
      <div className="analytics-stat-cards">
        <div className="stat-card">
          <div className="stat-value">{data.total_failures}</div>
          <div className="stat-label">Total Failures</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{Object.keys(data.strategy_counts).length}</div>
          <div className="stat-label">Strategies Used</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{Object.keys(data.query_counts).length}</div>
          <div className="stat-label">Unique Queries Failed</div>
        </div>
      </div>

      <div className="analytics-grid">
        <div className="analytics-card">
          <h3>Strategy Distribution</h3>
          <div className="bar-chart">
            {Object.entries(data.strategy_counts).map(([strategy, count]) => (
              <div key={strategy} className="bar-row">
                <span className="bar-label">{strategy}</span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${(count / maxStrategyCount) * 100}%`,
                      background: strategyColor(strategy),
                    }}
                  />
                </div>
                <span className="bar-count">{count}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="analytics-card">
          <h3>Failure Reasons</h3>
          <div className="bar-chart">
            {Object.entries(data.reason_counts).map(([reason, count]) => (
              <div key={reason} className="bar-row">
                <span className="bar-label">{reason.replace(/_/g, " ")}</span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${(count / maxReasonCount) * 100}%`,
                      background: "#818cf8",
                    }}
                  />
                </div>
                <span className="bar-count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {topQueries.length > 0 && (
        <div className="analytics-card">
          <h3>Top Problem Queries</h3>
          <div className="problem-queries">
            {topQueries.map((item, i) => (
              <div key={i} className="problem-query-row">
                <span className="pq-rank">#{i + 1}</span>
                <span className="pq-query">{item.query}</span>
                <span className="pq-count">{item.count}×</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Document Manager ─────────────────────────────────────────────────────────

function DocumentManager({ onUploadSuccess }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState(null);
  const fileInputRef = useRef(null);

  const fetchDocs = useCallback(() => {
    setLoading(true);
    fetch(`${API_BASE}/documents`)
      .then((r) => r.json())
      .then((data) => setDocuments(data.documents || []))
      .catch(() => setDocuments([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  const handleUpload = async (file) => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (data.status === "success") {
        onUploadSuccess(data.message);
        fetchDocs();
      } else {
        onUploadSuccess("Error: " + data.message, "error");
      }
    } catch (e) {
      onUploadSuccess("Upload failed: " + e.message, "error");
    }
    setUploading(false);
  };

  const handleDelete = async (docName) => {
    setDeletingDoc(docName);
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docName)}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (data.status === "success") {
        onUploadSuccess(`Deleted '${docName}' from knowledge base.`, "success");
        fetchDocs();
      } else {
        onUploadSuccess(data.detail || "Delete failed.", "error");
      }
    } catch (e) {
      onUploadSuccess("Delete failed: " + e.message, "error");
    }
    setDeletingDoc(null);
  };

  const formatSize = (bytes) => {
    if (!bytes) return "—";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="doc-manager">
      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragOver ? "drag-over" : ""} ${uploading ? "uploading-zone" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) handleUpload(file);
        }}
        onClick={() => !uploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          style={{ display: "none" }}
          accept=".pdf,.txt,.docx"
          onChange={(e) => handleUpload(e.target.files[0])}
        />
        {uploading ? (
          <>
            <div className="spinner lg" />
            <p>Ingesting document...</p>
          </>
        ) : (
          <>
            <div className="drop-icon">📄</div>
            <p className="drop-title">Drop a document here</p>
            <p className="drop-sub">or click to browse — PDF, TXT, DOCX</p>
          </>
        )}
      </div>

      {/* Document List */}
      <div className="doc-list-header">
        <h3>Knowledge Base ({documents.length} document{documents.length !== 1 ? "s" : ""})</h3>
        <button className="refresh-btn" onClick={fetchDocs} title="Refresh">⟳</button>
      </div>

      {loading ? (
        <div className="doc-loading"><div className="spinner" /></div>
      ) : documents.length === 0 ? (
        <div className="doc-empty">
          <p>No documents ingested yet. Upload a PDF to get started.</p>
        </div>
      ) : (
        <div className="doc-list">
          {documents.map((doc) => (
            <div key={doc.name} className="doc-card">
              <div className="doc-icon">📄</div>
              <div className="doc-info">
                <div className="doc-name">{doc.name}</div>
                <div className="doc-meta">
                  <span className="doc-chip">{doc.chunk_count} chunks</span>
                  <span className="doc-chip">{formatSize(doc.size_bytes)}</span>
                </div>
              </div>
              <button
                className="doc-delete-btn"
                onClick={() => handleDelete(doc.name)}
                disabled={deletingDoc === doc.name}
                title="Remove from knowledge base"
              >
                {deletingDoc === doc.name ? <div className="spinner tiny" /> : "✕"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Chat Interface ────────────────────────────────────────────────────────────

function ChatInterface({ onUploadTrigger }) {
  const STORAGE_KEY = "synapse_chat_messages";
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    } catch {
      return [];
    }
  });
  const [loading, setLoading] = useState(false);
  const [currentNode, setCurrentNode] = useState(null);
  const [graphPath, setGraphPath] = useState([]);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-50)));
    } catch {
      // quota exceeded — ignore
    }
  }, [messages]);

  const clearChat = async () => {
    setMessages([]);
    setGraphPath([]);
    setCurrentNode(null);
    try {
      await fetch(`${API_BASE}/chat/reset`, { method: "POST" });
    } catch {
      // ignore network errors
    }
  };

  const askQuestion = async () => {
    if (!question.trim() || loading) return;
    const userQ = question.trim();
    setQuestion("");
    setLoading(true);
    setCurrentNode("analyze_query");
    setGraphPath([]);

    setMessages((prev) => [...prev, { role: "user", content: userQ }]);

    let botMsg = { role: "assistant", content: "", sources: [], metadata: null };
    setMessages((prev) => [...prev, { ...botMsg }]);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userQ }),
      });
      if (!res.ok) throw new Error("Backend connection failed.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const chunk = buffer.substring(0, boundary);
          buffer = buffer.substring(boundary + 2);
          if (chunk.startsWith("data: ")) {
            const dataStr = chunk.substring(6);
            if (dataStr === "[DONE]") break;
            try {
              const data = JSON.parse(dataStr);
              if (data.type === "metadata") {
                botMsg = { ...botMsg, sources: data.sources, metadata: { strategy: data.strategy, heals: data.heals, confidence: data.confidence } };
              } else if (data.type === "chunk") {
                botMsg = { ...botMsg, content: botMsg.content + data.text };
              } else if (data.type === "node") {
                setCurrentNode(data.current_node);
                setGraphPath((prev) => {
                  if (prev[prev.length - 1] !== data.current_node) return [...prev, data.current_node];
                  return prev;
                });
              }
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = { ...botMsg };
                return next;
              });
            } catch { /* json parse errors on partial frames */ }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: `⚠️ ${err.message}`, sources: [], metadata: null };
        return next;
      });
    }

    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const SUGGESTED = [
    "Summarize the key points from my document",
    "Who are the main entities mentioned?",
    "analyze failures",
  ];

  return (
    <div className="chat-layout">
      {/* Main Chat Area */}
      <div className="chat-container">
        {/* Header */}
        <div className="chat-header">
          <div className="header-left">
            <div className="status-dot" />
            <h1>Synapse AI</h1>
          </div>
          <div className="header-actions">
            <button className="icon-btn" onClick={onUploadTrigger} title="Upload Document">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
              </svg>
            </button>
            {messages.length > 0 && (
              <button className="icon-btn" onClick={clearChat} title="Clear Chat">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4h6v2" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="chat-box">
          {messages.length === 0 && !loading && (
            <div className="chat-welcome">
              <div className="welcome-icon">
                <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a5 5 0 1 1 0 10A5 5 0 0 1 12 2z" /><path d="M20 21a8 8 0 1 0-16 0" />
                  <circle cx="18" cy="5" r="3" fill="var(--primary)" stroke="none" />
                </svg>
              </div>
              <h2>How can I help you today?</h2>
              <p>Upload a document, then ask anything about it.</p>
              <div className="suggestions">
                {SUGGESTED.map((s) => (
                  <button key={s} className="suggestion-chip" onClick={() => { setQuestion(s); }}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message-wrapper ${msg.role}`}>
              {msg.role === "assistant" && (
                <div className="avatar assistant-avatar">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z" />
                  </svg>
                </div>
              )}
              <div className={`message ${msg.role}`}>
                <div className="message-content">
                  {msg.role === "assistant"
                    ? renderMarkdown(msg.content)
                    : <p>{msg.content}</p>
                  }
                </div>

                {/* Sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="message-sources">
                    <span className="sources-label">Sources</span>
                    {msg.sources.map((src, i) => (
                      <span key={i} className="source-tag">
                        <span className="source-num">{i + 1}</span> {src}
                      </span>
                    ))}
                  </div>
                )}

                {/* Metadata panel */}
                {msg.metadata && (
                  <div className="debug-panel">
                    <div className="debug-item">
                      <span className="debug-label">Strategy</span>
                      <span className="debug-val strategy" style={{ color: strategyColor(msg.metadata.strategy) }}>
                        {strategyIcon(msg.metadata.strategy)} {msg.metadata.strategy}
                      </span>
                    </div>
                    <div className="debug-item">
                      <span className="debug-label">Confidence</span>
                      <div className="confidence-bar-wrap">
                        <div className="confidence-bar">
                          <div
                            className="confidence-fill"
                            style={{
                              width: `${(msg.metadata.confidence * 100).toFixed(0)}%`,
                              background: msg.metadata.confidence > 0.6 ? "#1dd4a9" : msg.metadata.confidence > 0.3 ? "#f59e0b" : "#f87171",
                            }}
                          />
                        </div>
                        <span>{(msg.metadata.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    {msg.metadata.heals > 0 && (
                      <div className="debug-item">
                        <span className="debug-label">Self-Heals</span>
                        <span className="debug-val heal-badge">🔁 {msg.metadata.heals}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-wrapper assistant">
              <div className="avatar assistant-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z" />
                </svg>
              </div>
              <div className="message assistant">
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div className="input-area">
          <div className="input-box">
            <textarea
              placeholder="Message Synapse AI... (Enter to send, Shift+Enter for newline)"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={askQuestion}
              disabled={loading || !question.trim()}
            >
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
          <p className="input-hint">Synapse AI can make mistakes. Verify important information.</p>
        </div>
      </div>

      {/* LangGraph Sidebar */}
      <div className="graph-sidebar">
        <div className="graph-header">
          <h3>Pipeline</h3>
          {loading && <span className="live-badge">LIVE</span>}
        </div>
        <div className="graph-nodes">
          {GRAPH_NODES.map((node) => {
            const isActive = currentNode === node;
            const isVisited = graphPath.includes(node);
            return (
              <div key={node} className={`graph-node ${isActive ? "active" : ""} ${isVisited && !isActive ? "visited" : ""}`}>
                <div className="node-dot">
                  {isVisited && !isActive ? (
                    <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : isActive ? (
                    <div className="pulse-dot" />
                  ) : null}
                </div>
                <div className="node-label">
                  {node.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                </div>
              </div>
            );
          })}
        </div>
        {graphPath.length > 0 && (
          <div className="graph-path-log">
            <strong>Execution Path</strong>
            <div className="path-text">{graphPath.join(" → ")}</div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Root App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = "success") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <div className="app">
      {/* Tab navigation */}
      <nav className="tab-nav">
        <div className="nav-brand">
          <div className="nav-logo">⚡</div>
          <span>Synapse AI</span>
        </div>
        <div className="nav-tabs">
          {[
            { id: "chat", label: "Chat", icon: "💬" },
            { id: "documents", label: "Documents", icon: "📄" },
            { id: "analytics", label: "Analytics", icon: "📊" },
          ].map((tab) => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="nav-tab-icon">{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Page Content */}
      <main className="app-main">
        {activeTab === "chat" && (
          <ChatInterface onUploadTrigger={() => setActiveTab("documents")} />
        )}
        {activeTab === "documents" && (
          <div className="page-container">
            <div className="page-header">
              <h2>Knowledge Base</h2>
              <p>Manage the documents your AI assistant can search through.</p>
            </div>
            <DocumentManager onUploadSuccess={(msg, type) => { addToast(msg, type); }} />
          </div>
        )}
        {activeTab === "analytics" && (
          <div className="page-container">
            <div className="page-header">
              <h2>Failure Analytics</h2>
              <p>Monitor how your RAG pipeline's self-healing strategies perform over time.</p>
            </div>
            <AnalyticsDashboard />
          </div>
        )}
      </main>

      {/* Toast notifications */}
      <div className="toast-container">
        {toasts.map((t) => (
          <Toast key={t.id} message={t.message} type={t.type} onClose={() => removeToast(t.id)} />
        ))}
      </div>
    </div>
  );
}