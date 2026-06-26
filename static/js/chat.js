// Customer Chat JavaScript
class CustomerChat {
    constructor() {
        this.socket = null;
        this.conversationId = null;
        this.customerName = null;
        this.isConnected = false;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        this.connectionForm = document.getElementById('connection-form');
        this.chatInterface = document.getElementById('chat-interface');
        this.startChatForm = document.getElementById('start-chat-form');
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');
    }
    
    bindEvents() {
        this.startChatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.startConversation();
        });
        
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && this.isConnected) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.sendButton.addEventListener('click', () => {
            if (this.isConnected) {
                this.sendMessage();
            }
        });
    }
    
    startConversation() {
        const customerName = document.getElementById('customer-name').value.trim();
        const customerEmail = document.getElementById('customer-email').value.trim();
        const subject = document.getElementById('subject').value.trim();
        
        if (!customerName || !customerEmail || !subject) {
            this.showError('Please fill in all fields');
            return;
        }
        
        this.customerName = customerName;
        
        // Initialize Socket.IO connection
        this.socket = io();
        
        // Socket event handlers
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateStatus('connecting', 'Connecting to support...');
            
            // Start the conversation
            this.socket.emit('start_conversation', {
                customer_name: customerName,
                customer_email: customerEmail,
                subject: subject
            });
        });
        
        this.socket.on('conversation_started', (data) => {
            console.log('Conversation started:', data);
            this.conversationId = data.conversation_id;
            this.isConnected = true;
            
            // Switch to chat interface
            this.connectionForm.style.display = 'none';
            this.chatInterface.style.display = 'block';
            
            // Enable input and button
            this.messageInput.disabled = false;
            this.sendButton.disabled = false;
            
            // Update status
            this.updateStatus('connected', 'Connected to support');
            
            // Clear messages and show welcome message
            this.chatMessages.innerHTML = '';
            this.addMessage('system', 'ConvoSense', data.message);
            
            // Focus on message input
            this.messageInput.focus();
        });
        
        this.socket.on('new_message', (data) => {
            console.log('New message:', data);
            this.addMessage(data.sender_type, data.sender_name, data.content, data.timestamp, data.detected_intents);
        });
        
        this.socket.on('agent_joined', (data) => {
            console.log('Agent joined:', data);
            this.addMessage('system', 'ConvoSense', data.message);
        });
        
        this.socket.on('conversation_status_updated', (data) => {
            console.log('Conversation status updated:', data);
            this.addMessage('system', 'ConvoSense', `Conversation ${data.status}`);
            
            if (data.status === 'closed') {
                this.messageInput.disabled = true;
                this.sendButton.disabled = true;
                this.updateStatus('disconnected', 'Conversation closed');
            }
        });
        
        this.socket.on('error', (data) => {
            console.error('Socket error:', data);
            this.showError(data.message || 'Connection error occurred');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            this.updateStatus('disconnected', 'Connection lost');
            this.messageInput.disabled = true;
            this.sendButton.disabled = true;
        });
    }
    
    sendMessage() {
        const content = this.messageInput.value.trim();
        if (!content || !this.isConnected) return;
        
        this.socket.emit('send_message', {
            conversation_id: this.conversationId,
            sender_type: 'customer',
            sender_name: this.customerName,
            content: content
        });
        
        this.messageInput.value = '';
    }
    
    addMessage(senderType, senderName, content, timestamp = null, intents = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${senderType}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.textContent = content;
        
        const infoDiv = document.createElement('div');
        infoDiv.className = 'message-info';
        
        if (timestamp) {
            const time = new Date(timestamp).toLocaleTimeString();
            infoDiv.textContent = `${senderName} • ${time}`;
        } else {
            const time = new Date().toLocaleTimeString();
            infoDiv.textContent = `${senderName} • ${time}`;
        }
        
        messageDiv.appendChild(bubbleDiv);
        messageDiv.appendChild(infoDiv);
        
        // Add intent information for customer messages
        if (intents && senderType === 'customer') {
            const intentDiv = document.createElement('div');
            intentDiv.className = 'intent-info mt-2';
            intentDiv.innerHTML = this.formatIntents(intents);
            messageDiv.appendChild(intentDiv);
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatIntents(intentData) {
        if (!intentData || !intentData.intents) return '';
        
        let html = '<small class="text-muted"><i data-feather="cpu"></i> AI Analysis: ';
        
        if (intentData.primary_intent) {
            html += `<span class="badge bg-primary">${intentData.primary_intent}</span> `;
        }
        
        if (intentData.sentiment) {
            const sentimentColor = {
                'positive': 'success',
                'neutral': 'secondary',
                'negative': 'danger'
            }[intentData.sentiment] || 'secondary';
            html += `<span class="badge bg-${sentimentColor}">${intentData.sentiment}</span> `;
        }
        
        if (intentData.urgency) {
            const urgencyColor = {
                'high': 'danger',
                'medium': 'warning',
                'low': 'success'
            }[intentData.urgency] || 'secondary';
            html += `<span class="badge bg-${urgencyColor}">${intentData.urgency} priority</span>`;
        }
        
        html += '</small>';
        return html;
    }
    
    updateStatus(status, text) {
        const statusClasses = {
            'connecting': 'bg-warning',
            'connected': 'bg-success',
            'disconnected': 'bg-danger'
        };
        
        this.statusIndicator.className = `badge ${statusClasses[status] || 'bg-secondary'}`;
        this.statusIndicator.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        this.statusText.textContent = text;
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showError(message) {
        // Create error alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at top of container
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CustomerChat();
    
    // Refresh feather icons after dynamic content
    const observer = new MutationObserver(() => {
        feather.replace();
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
