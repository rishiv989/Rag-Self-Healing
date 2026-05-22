import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [currentNode, setCurrentNode] = useState(null);
  const [graphPath, setGraphPath] = useState([]);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(`Ingesting ${file.name}... (Semantic Chunking)`);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (data.status === "success") {
        setUploadProgress(data.message);
        setTimeout(() => setUploadProgress(""), 5000);
      } else {
        setUploadProgress("Error: " + data.message);
      }
    } catch (err) {
      setUploadProgress("Upload failed: " + err.message);
    }
    
    setUploading(false);
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    const userMessage = {
      role: "user",
      content: question
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);
    setCurrentNode("analyze_query");
    setGraphPath([]);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
      });

      if (!response.ok) {
        throw new Error("Backend connection failed.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let botMessage = {
        role: "assistant",
        content: "",
        sources: [],
        metadata: null
      };

      setMessages((prev) => [...prev, botMessage]);

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
                botMessage.sources = data.sources;
                botMessage.metadata = {
                  strategy: data.strategy,
                  heals: data.heals,
                  confidence: data.confidence
                };
              } else if (data.type === "chunk") {
                botMessage.content += data.text;
              } else if (data.type === "node") {
                setCurrentNode(data.current_node);
                setGraphPath(prev => {
                  if (prev[prev.length - 1] !== data.current_node) {
                    return [...prev, data.current_node];
                  }
                  return prev;
                });
              }
              
              setMessages((prev) => {
                const newMsgs = [...prev];
                newMsgs[newMsgs.length - 1] = { ...botMessage };
                return newMsgs;
              });
            } catch (e) {
              console.error("JSON parse error on streaming chunk", e);
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${error.message}`
        }
      ]);
    }

    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      askQuestion();
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        
        <div className="chat-header">
          <div className="header-left">
            <div className="status-dot"></div>
            <h1>Synapse AI</h1>
          </div>
        </div>
        
        {uploadProgress && (
          <div className="upload-toast">
            {uploading && <div className="spinner small"></div>}
            <span>{uploadProgress}</span>
          </div>
        )}

        <div className="chat-box">
          {messages.length === 0 && !loading && (
            <div className="message-wrapper assistant" style={{ alignItems: 'center', opacity: 0.6, marginTop: 'auto', marginBottom: 'auto' }}>
              <div style={{ textAlign: 'center' }}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '16px', stroke: 'var(--accent)' }}>
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                </svg>
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '20px', fontWeight: '500', color: 'var(--text-main)', margin: '0 0 8px 0' }}>How can I help you today?</h2>
                <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>Ask me anything. I'll search my knowledge base.</p>
              </div>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message-wrapper ${msg.role}`}>
              <div className={`message ${msg.role}`}>
                <div className="message-content">
                  {msg.content.split(/(\[\d+\])/g).map((part, i) => {
                    if (part.match(/\[\d+\]/)) {
                      const citationNum = part.replace(/[\[\]]/g, '');
                      return (
                        <span key={i} className="citation-badge" title={msg.sources ? msg.sources[citationNum - 1] : "Source"}>
                          {citationNum}
                        </span>
                      );
                    }
                    return <span key={i}>{part}</span>;
                  })}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="message-sources">
                    <strong>References:</strong>
                    <ul>
                      {msg.sources.map((src, i) => (
                        <li key={i}><span className="citation-badge">{i + 1}</span> {src}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {msg.metadata && (
                  <div className="debug-panel">
                    <div className="debug-item">
                      <span className="debug-label">Strategy</span>
                      <span className="debug-val strategy">{msg.metadata.strategy}</span>
                    </div>
                    <div className="debug-item">
                      <span className="debug-label">Confidence</span>
                      <span className="debug-val">{(msg.metadata.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="debug-item">
                      <span className="debug-label">Heals</span>
                      <span className="debug-val">{msg.metadata.heals}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-wrapper assistant">
              <div className="message assistant">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <div className="input-box">
            <label className={`upload-btn ${uploading ? 'uploading' : ''}`} title="Upload Document">
               <input type="file" style={{display: 'none'}} onChange={handleFileUpload} accept=".pdf,.txt,.docx" disabled={uploading}/>
               {uploading ? (
                 <div className="spinner"></div>
               ) : (
                 <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                 </svg>
               )}
            </label>
            <input
              type="text"
              placeholder="Message Synapse AI..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button className="send-btn" onClick={askQuestion} disabled={loading || !question.trim()}>
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
              </svg>
            </button>
          </div>
        </div>

      </div>
      
      {/* LangGraph Sidebar */}
      <div className="graph-sidebar">
        <div className="graph-header">
          <h3>State Machine</h3>
          <span className="live-badge">LIVE</span>
        </div>
        <div className="graph-nodes">
          {['analyze_query', 'check_fast_memory', 'retrieve_and_rerank', 'self_heal', 'web_search_fallback', 'build_knowledge_graph', 'generate_draft', 'reflect'].map((node) => {
            const isActive = currentNode === node;
            const isVisited = graphPath.includes(node);
            return (
              <div key={node} className={`graph-node ${isActive ? 'active' : ''} ${isVisited && !isActive ? 'visited' : ''}`}>
                <div className="node-icon">
                   {isVisited && !isActive ? (
                     <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>
                   ) : isActive ? (
                     <div className="pulse-dot"></div>
                   ) : null}
                </div>
                <div className="node-label">
                  {node.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
              </div>
            );
          })}
        </div>
        <div className="graph-path-log">
           <strong>Execution Path:</strong>
           <div className="path-text">
             {graphPath.length > 0 ? graphPath.join(" → ") : "Awaiting input..."}
           </div>
        </div>
      </div>
    </div>
  );
}

export default App;