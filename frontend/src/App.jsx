import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE = '/api'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [config, setConfig] = useState(null)
  const [showConfig, setShowConfig] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [documents, setDocuments] = useState([])
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    fetchConfig()
    fetchDocuments()
  }, [])

  const fetchConfig = async () => {
    try {
      const response = await axios.get(`${API_BASE}/config`)
      setConfig(response.data)
    } catch (error) {
      console.error('Error fetching config:', error)
    }
  }

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE}/documents`)
      setDocuments(response.data.documents || [])
    } catch (error) {
      console.error('Error fetching documents:', error)
    }
  }

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await axios.post(`${API_BASE}/query`, {
        question: input
      })

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        retrievedDocs: response.data.retrieved_docs || [],
        searchMode: response.data.search_mode || 'hybrid'
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage = {
        role: 'error',
        content: error.response?.data?.detail || 'An error occurred. Please try again.'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleConfigUpdate = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      await axios.post(`${API_BASE}/config`, config)
      alert('Configuration updated successfully!')
    } catch (error) {
      alert('Error updating configuration: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setLoading(true)
    try {
      await axios.post(`${API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      alert('File uploaded successfully!')
      fetchDocuments()
    } catch (error) {
      alert('Error uploading file: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
      e.target.value = ''
    }
  }

  const handleRebuildVectorStore = async () => {
    setLoading(true)
    try {
      await axios.post(`${API_BASE}/rebuild-vectorstore`)
      alert('Vector store rebuilt successfully!')
    } catch (error) {
      alert('Error rebuilding vector store: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  return (
    <div className="app">
      <header className="header">
        <h1>FinanSearch - RAG Chatbot</h1>
        <div className="header-buttons">
          <button onClick={() => setShowUpload(!showUpload)} className="header-btn">
            {showUpload ? 'Hide Upload' : 'Upload Docs'}
          </button>
          <button onClick={() => setShowConfig(!showConfig)} className="header-btn">
            {showConfig ? 'Hide Config' : 'Show Config'}
          </button>
        </div>
      </header>

      <div className="main-container">
        {showConfig && config && (
          <div className="config-panel">
            <h2>RAG Configuration</h2>
            <form onSubmit={handleConfigUpdate}>
              <div className="config-section">
                <h3>Model Settings</h3>
                <div className="form-group">
                  <label>LLM Model</label>
                  <select
                    value={config.llm_model}
                    onChange={(e) => setConfig({...config, llm_model: e.target.value})}
                  >
                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Embedding Model</label>
                  <select
                    value={config.embedding_model}
                    onChange={(e) => setConfig({...config, embedding_model: e.target.value})}
                  >
                    <option value="text-embedding-3-small">text-embedding-3-small</option>
                    <option value="text-embedding-3-large">text-embedding-3-large</option>
                    <option value="text-embedding-ada-002">text-embedding-ada-002</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Temperature: {config.temperature}</label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={config.temperature}
                    onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value)})}
                  />
                </div>
              </div>

              <div className="config-section">
                <h3>Chunking Parameters</h3>
                <div className="form-group">
                  <label>Chunk Size</label>
                  <input
                    type="number"
                    value={config.chunk_size}
                    onChange={(e) => setConfig({...config, chunk_size: parseInt(e.target.value)})}
                  />
                </div>

                <div className="form-group">
                  <label>Chunk Overlap</label>
                  <input
                    type="number"
                    value={config.chunk_overlap}
                    onChange={(e) => setConfig({...config, chunk_overlap: parseInt(e.target.value)})}
                  />
                </div>

                <div className="form-group">
                  <label>Separator</label>
                  <input
                    type="text"
                    value={config.separator}
                    onChange={(e) => setConfig({...config, separator: e.target.value})}
                    placeholder="\n\n"
                  />
                </div>
              </div>

              <div className="config-section">
                <h3>Retrieval Settings</h3>
                <div className="form-group">
                  <label>Search Mode</label>
                  <select
                    value={config.search_mode || 'hybrid'}
                    onChange={(e) => setConfig({...config, search_mode: e.target.value})}
                  >
                    <option value="hybrid">Hybrid (Keyword + Semantic)</option>
                    <option value="semantic">Semantic Only</option>
                    <option value="keyword">Keyword Only (BM25)</option>
                  </select>
                  <small className="help-text">
                    {config.search_mode === 'hybrid' && 'Combines keyword matching with semantic understanding'}
                    {config.search_mode === 'semantic' && 'Uses AI embeddings to understand meaning'}
                    {config.search_mode === 'keyword' && 'Uses BM25 algorithm for exact term matching'}
                  </small>
                </div>

                <div className="form-group">
                  <label>Number of Retrieved Documents: {config.retrieval_k}</label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={config.retrieval_k}
                    onChange={(e) => setConfig({...config, retrieval_k: parseInt(e.target.value)})}
                  />
                </div>
              </div>

              <div className="config-section">
                <h3>Prompt Template</h3>
                <div className="form-group">
                  <textarea
                    value={config.prompt_template}
                    onChange={(e) => setConfig({...config, prompt_template: e.target.value})}
                    rows="8"
                  />
                </div>
              </div>

              <button type="submit" className="save-btn" disabled={loading}>
                {loading ? 'Saving...' : 'Save Configuration'}
              </button>
            </form>
          </div>
        )}

        {showUpload && (
          <div className="upload-panel">
            <h2>Document Management</h2>

            <div className="upload-section">
              <div className="form-group">
                <label className="file-upload-label">
                  <input
                    type="file"
                    onChange={handleFileUpload}
                    accept=".txt,.pdf,.md"
                  />
                  Choose File (.txt, .pdf, .md)
                </label>
              </div>

              <button
                onClick={handleRebuildVectorStore}
                className="rebuild-btn"
                disabled={loading}
              >
                {loading ? 'Rebuilding...' : 'Rebuild Vector Store'}
              </button>
            </div>

            <div className="documents-list">
              <h3>Uploaded Documents ({documents.length})</h3>
              {documents.length === 0 ? (
                <p className="no-docs">No documents uploaded yet</p>
              ) : (
                <ul>
                  {documents.map((doc, index) => (
                    <li key={index}>
                      <span className="doc-name">{doc.name}</span>
                      <span className="doc-size">{formatFileSize(doc.size)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}

        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>Welcome to FinanSearch!</h2>
                <p>Ask questions about your uploaded documents.</p>
                <p>Upload documents and configure RAG parameters using the buttons above.</p>
              </div>
            )}
            {messages.map((message, index) => (
              <div key={index} className={`message-wrapper ${message.role}`}>
                <div className={`message ${message.role}`}>
                  <div className="message-content">
                    {message.content}
                  </div>
                </div>
                {message.role === 'assistant' && message.retrievedDocs && message.retrievedDocs.length > 0 && (
                  <div className="retrieved-docs">
                    <div className="retrieved-header">
                      <span className="search-mode-badge">{message.searchMode || 'hybrid'} search</span>
                      <span>Retrieved {message.retrievedDocs.length} relevant chunks:</span>
                    </div>
                    {message.retrievedDocs.map((doc, docIndex) => (
                      <div key={docIndex} className="retrieved-doc">
                        <div className="doc-header">
                          <span className="doc-source">{doc.source}</span>
                          <span className="doc-score">Score: {doc.score}</span>
                        </div>
                        <div className="doc-content">{doc.content}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <div className="message-content typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSendMessage} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              disabled={loading}
            />
            <button type="submit" disabled={loading || !input.trim()}>
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App
