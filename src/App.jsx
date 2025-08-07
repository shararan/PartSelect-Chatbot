import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Search, ExternalLink, DollarSign, Star } from 'lucide-react';
import './App.css'; // Import our custom CSS

const API_BASE_URL = 'http://127.0.0.1:5000';

// PartSelect Logo Component
const PartSelectLogo = ({ className = "" }) => (
  <div className={`logo-container ${className}`}>
    <img 
      src="../public/partselect_logo.jpeg" 
      alt="PartSelect" 
      className="logo-image"
      onError={(e) => {
        e.target.style.display = 'none';
        e.target.nextSibling.style.display = 'flex';
      }}
    />
    {/* Fallback logo */}
    <div className="logo-fallback">
      <div className="logo-icon">P</div>
      <div className="logo-text">
        <span>Part</span><span>Select</span>
      </div>
    </div>
  </div>
);

const PartSelectChat = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm your PartSelect assistant. I can help you find dishwasher parts, answer questions about installation, pricing, and compatibility. Try asking me about a specific part number or describe the problem you're having!",
      timestamp: new Date().toLocaleTimeString(),
      sources: []
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check API connection on mount
  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        setIsConnected(true);
      }
    } catch (error) {
      console.error('API connection failed:', error);
      setIsConnected(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !isConnected) return;

    const userMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString(),
      sources: []
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          conversation_history: messages.slice(-10) // Send last 10 messages for context
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toLocaleTimeString(),
        sources: data.sources || [],
        num_sources_found: data.num_sources_found || 0
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request. Please make sure the backend server is running and try again.',
        timestamp: new Date().toLocaleTimeString(),
        sources: []
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const PartCard = ({ part }) => {
    const handleUrlClick = () => {
      if (part.url && part.url !== 'N/A') {
        window.open(part.url, '_blank');
      }
    };
  
    return (
      <div className="part-card">
        <div className="part-card-header">
          <h4 className="part-name">{part.part_name}</h4>
          <div className="similarity-score">
            <Star size={12} />
            {(part.similarity_score * 100).toFixed(0)}%
          </div>
        </div>
        
        <div className="part-details">
          <div className="detail-row">
            <span className="label">Part ID:</span>
            <span className="value part-id">{part.part_id}</span>
          </div>
          
          <div className="detail-row">
            <span className="label">Brand:</span>
            <span className="value">{part.brand}</span>
          </div>
          
          {part.price !== 'N/A' && (
            <div className="detail-row">
              <span className="label">Price:</span>
              <span className="value price">
                <DollarSign size={12} />
                {part.price}
              </span>
            </div>
          )}
          
          {/* REMOVED: Fixes section */}
          
          {part.url !== 'N/A' && (
            <button onClick={handleUrlClick} className="view-details-btn">
              <ExternalLink size={12} />
              View on PartSelect
            </button>
          )}
        </div>
      </div>
    );
  };

  const QuickActions = () => {
    const quickQueries = [
      "What parts fix a dishwasher that won't drain?",
      "Show me Whirlpool dishwasher parts under $50",
      "PS11752778 installation instructions",
      "Compatible parts for WDT780SAEM1"
    ];

    return (
      <div className="quick-actions">
        <p className="quick-actions-label">Try these common questions:</p>
        <div className="quick-buttons">
          {quickQueries.map((query, index) => (
            <button
              key={index}
              onClick={() => setInputMessage(query)}
              className="quick-button"
            >
              {query}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const ConnectionStatus = () => (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <div className="status-dot"></div>
      {isConnected ? 'Connected' : 'Disconnected'}
    </div>
  );

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="header">
        <div className="header-content">
          <div className="header-left">
            <PartSelectLogo />
            <div className="header-info">
              <h2 className="header-title">Chat Assistant</h2>
              <p className="header-subtitle">Find dishwasher parts, installation help, and more</p>
            </div>
          </div>
          <div className="header-right">
            <ConnectionStatus />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {!isConnected && (
          <div className="connection-warning">
            <div className="warning-content">
              <div className="warning-icon">⚠️</div>
              <div>
                <p className="warning-title">
                  <strong>API Not Connected:</strong> Please start the Python backend server first.
                </p>
                <p className="warning-subtitle">
                  Run: <code>python rag_api.py</code>
                </p>
              </div>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-content">
              <div className="message-avatar">
                {message.role === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              
              <div className="message-body">
                <div className="message-bubble">
                  <div className="message-text">{message.content}</div>
                </div>
                
                <div className="message-meta">
                  {message.timestamp}
                  {message.num_sources_found > 0 && (
                    <span> • {message.num_sources_found} sources found</span>
                  )}
                </div>
                
                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                  <div className="sources-container">
                    <div className="sources-header">
                      <Search size={16} />
                      Related Parts ({message.sources.length})
                    </div>
                    <div className="sources-grid">
                      {message.sources.map((source, sourceIndex) => (
                        <PartCard key={sourceIndex} part={source} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="message-avatar">
                <Bot size={16} />
              </div>
              <div className="message-body">
                <div className="message-bubble">
                  <div className="typing-indicator">
                    <div className="dot"></div>
                    <div className="dot"></div>
                    <div className="dot"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-area">
        {messages.length <= 1 && <QuickActions />}
        
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isConnected ? "Ask about dishwasher parts, installation, or enter a part number..." : "Please start the backend server first..."}
              className="message-input"
              rows={2}
              disabled={isLoading || !isConnected}
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading || !isConnected}
            className="send-button"
          >
            <Send size={20} />
          </button>
        </div>
        
        <div className="input-help">
          Press Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};

export default PartSelectChat;