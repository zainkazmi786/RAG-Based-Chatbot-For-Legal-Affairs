// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const clearChatBtn = document.getElementById('clear-chat');
    const docCountEl = document.getElementById('doc-count');
    const addSourceBtn = document.getElementById('add-source-btn');
    const sourcePathInput = document.getElementById('source-path');
    const sourceAddMessage = document.getElementById('source-add-message');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');
    
    // Chat history
    let chatHistory = [];
    
    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        // Reset height to auto to get correct scrollHeight
        this.style.height = 'auto';
        // Set height to scrollHeight + border
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Get document count
    fetchDocumentCount();
    
    // Set up event listeners
    chatForm.addEventListener('submit', handleChatSubmit);
    clearChatBtn.addEventListener('click', clearChat);
    addSourceBtn.addEventListener('click', addDataSource);
    
    // Set up suggestion chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            userInput.value = chip.textContent;
            userInput.dispatchEvent(new Event('input')); // Trigger resize
            userInput.focus();
        });
    });
    
    /**
     * Handle chat form submission
     * @param {Event} e - Form submit event
     */
    function handleChatSubmit(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input and reset height
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to server
        sendMessageToServer(message);
    }
    
    /**
     * Add a message to the chat interface
     * @param {string} role - 'user', 'assistant', or 'system'
     * @param {string} content - Message content
     * @param {Object} [options] - Additional options
     */
    function addMessageToChat(role, content, options = {}) {
        // Create message container
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        // Create message content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Parse markdown for assistant messages
        if (role === 'assistant' || role === 'system') {
            contentDiv.innerHTML = marked.parse(content);
        } else {
            contentDiv.textContent = content;
        }
        
        // Add timestamp if provided
        if (options.timestamp) {
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'message-timestamp';
            timestampDiv.textContent = formatTimestamp(options.timestamp);
            contentDiv.appendChild(timestampDiv);
        }
        
        // Add sources info if provided
        if (options.sourcesCount) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message-sources';
            sourcesDiv.textContent = `Based on ${options.sourcesCount} legal sources`;
            contentDiv.appendChild(sourcesDiv);
        }
        
        // Add message to chat
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Add to chat history
        chatHistory.push({
            role,
            content,
            timestamp: options.timestamp || Date.now()
        });
    }
    
    /**
     * Send message to server
     * @param {string} message - User message
     */
    function sendMessageToServer(message) {
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add assistant message to chat
            addMessageToChat('assistant', data.response, {
                timestamp: data.timestamp,
                sourcesCount: data.sources_count
            });
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            addMessageToChat('system', `Error: ${error.message || 'Failed to get response'}`);
        });
    }
    
    /**
     * Show typing indicator
     */
    function showTypingIndicator() {
        typingIndicator.classList.add('active');
    }
    
    /**
     * Hide typing indicator
     */
    function hideTypingIndicator() {
        typingIndicator.classList.remove('active');
    }
    
    /**
     * Clear chat history
     */
    function clearChat() {
        const systemMessage = chatMessages.querySelector('.message.system');

        // Clear UI
        while (chatMessages.firstChild) {
            const firstChild = chatMessages.firstChild;
        
            // Don't delete the system message if it's the current first child
            if (firstChild === systemMessage) {
                chatMessages.removeChild(firstChild); // Temporarily remove it
                continue; // We'll add it back after clearing
            }
        
            chatMessages.removeChild(firstChild);
        }
        
        // Add system message back to top (if it existed)
        if (systemMessage) {
            chatMessages.appendChild(systemMessage);
        }
        // Clear chat history
        chatHistory = [];
    }
    
    /**
     * Add new data source
     */
    function addDataSource() {
        const sourcePath = sourcePathInput.value.trim();
        if (!sourcePath) return;
        
        // Clear previous messages
        sourceAddMessage.textContent = '';
        sourceAddMessage.className = 'message';
        
        // Send request
        fetch('/api/add-source', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ source_path: sourcePath })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Update document count
                if (data.document_count) {
                    docCountEl.textContent = data.document_count;
                }
                
                // Show success message
                sourceAddMessage.textContent = data.message;
                sourceAddMessage.className = 'message success';
                
                // Clear input
                sourcePathInput.value = '';
            } else {
                // Show warning message
                sourceAddMessage.textContent = data.message;
                sourceAddMessage.className = 'message warning';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            sourceAddMessage.textContent = error.message || 'Failed to add source';
            sourceAddMessage.className = 'message error';
        });
    }
    
    /**
     * Fetch document count
     */
    function fetchDocumentCount() {
        fetch('/api/document-count')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                docCountEl.textContent = data.count;
            })
            .catch(error => {
                console.error('Error:', error);
                docCountEl.textContent = 'Error loading';
            });
    }
    
    /**
     * Format timestamp
     * @param {number} timestamp - Unix timestamp
     * @returns {string} Formatted time string
     */
    function formatTimestamp(timestamp) {
        const date = new Date(timestamp * 1000);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
} })
