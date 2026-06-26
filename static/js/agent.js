// Agent Dashboard JavaScript
class AgentDashboard {
    constructor() {
        this.socket = null;
        this.currentConversationId = null;
        this.agentName = 'Support Agent';
        this.isConnectedAsAgent = false;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        this.conversationsList = document.getElementById('conversations-list');
        this.agentNameInput = document.getElementById('agent-name');
        this.joinAgentsButton = document.getElementById('join-agents');
        this.noConversationPanel = document.getElementById('no-conversation');
        this.chatInterface = document.getElementById('agent-chat-interface');
        this.chatMessages = document.getElementById('agent-chat-messages');
        this.messageInput = document.getElementById('agent-message-input');
        this.sendButton = document.getElementById('agent-send-button');
        this.conversationTitle = document.getElementById('current-conversation-title');
        this.conversationControls = document.getElementById('conversation-controls');
        this.closeButton = document.getElementById('close-conversation');
        this.reopenButton = document.getElementById('reopen-conversation');
        this.intentPanel = document.getElementById('intent-panel');
        this.intentResults = document.getElementById('intent-results');
    }
    
    bindEvents() {
        this.joinAgentsButton.addEventListener('click', () => {
            this.connectAsAgent();
        });
        
        // Conversation selection
        this.conversationsList.addEventListener('click', (e) => {
            const conversationItem = e.target.closest('.conversation-item');
            if (conversationItem) {
                this.selectConversation(conversationItem);
            }
        });
        
        // Message sending
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && this.currentConversationId) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.sendButton.addEventListener('click', () => {
            if (this.currentConversationId) {
                this.sendMessage();
            }
        });
        
        // Conversation controls
        this.closeButton.addEventListener('click', () => {
            this.updateConversationStatus('closed');
        });
        
        this.reopenButton.addEventListener('click', () => {
            this.updateConversationStatus('active');
        });
        
        // Agent name update
        this.agentNameInput.addEventListener('change', (e) => {
            this.agentName = e.target.value.trim() || 'Support Agent';
        });
    }
    
    connectAsAgent() {
        this.agentName = this.agentNameInput.value.trim() || 'Support Agent';
        
        // Initialize Socket.IO connection
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected as agent');
            
            // Join agent room for notifications
            this.socket.emit('join_agent_room', {
                agent_name: this.agentName
            });
        });
        
        this.socket.on('joined_agents', (data) => {
            console.log('Joined agents room:', data);
            this.isConnectedAsAgent = true;
            this.joinAgentsButton.textContent = 'Connected';
            this.joinAgentsButton.disabled = true;
            this.joinAgentsButton.className = 'btn btn-success w-100';
        });
        
        this.socket.on('new_conversation', (data) => {
            console.log('New conversation:', data);
            this.addNewConversationToList(data);
            this.showNotification(`New conversation from ${data.customer_name}`, 'info');
        });
        
        this.socket.on('new_message', (data) => {
            console.log('New message:', data);
            if (data.conversation_id == this.currentConversationId) {
                this.addMessage(data);
                
                // Show intent analysis for customer messages
                if (data.sender_type === 'customer' && data.detected_intents) {
                    this.showIntentAnalysis(data.detected_intents, data.message_id);
                }
            }
            
            // Update conversation in list
            this.updateConversationInList(data.conversation_id);
        });
        
        this.socket.on('agent_joined', (data) => {
            console.log('Agent joined:', data);
            if (this.currentConversationId) {
                this.addMessage({
                    sender_type: 'system',
                    sender_name: 'ConvoSense',
                    content: data.message,
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        this.socket.on('conversation_status_updated', (data) => {
            console.log('Conversation status updated:', data);
            if (data.conversation_id == this.currentConversationId) {
                this.updateConversationUI(data.status);
                this.addMessage({
                    sender_type: 'system',
                    sender_name: 'ConvoSense',
                    content: `Conversation ${data.status} by ${data.agent_name}`,
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        this.socket.on('error', (data) => {
            console.error('Socket error:', data);
            this.showNotification(data.message || 'Connection error occurred', 'danger');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnectedAsAgent = false;
            this.joinAgentsButton.textContent = 'Reconnect';
            this.joinAgentsButton.disabled = false;
            this.joinAgentsButton.className = 'btn btn-warning w-100';
        });
    }
    
    selectConversation(conversationItem) {
        // Remove active class from all items
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to selected item
        conversationItem.classList.add('active');
        
        const conversationId = conversationItem.dataset.conversationId;
        const customerName = conversationItem.dataset.customerName;
        const subject = conversationItem.dataset.subject;
        
        this.currentConversationId = conversationId;
        
        // Update UI
        this.conversationTitle.textContent = `${customerName} - ${subject}`;
        this.noConversationPanel.style.display = 'none';
        this.chatInterface.style.display = 'block';
        this.conversationControls.style.display = 'block';
        
        // Enable message input
        this.messageInput.disabled = false;
        this.sendButton.disabled = false;
        
        // Load conversation messages
        this.loadConversation(conversationId);
        
        // Join conversation room
        if (this.socket && this.isConnectedAsAgent) {
            this.socket.emit('join_conversation', {
                conversation_id: conversationId,
                agent_name: this.agentName
            });
        }
    }
    
    async loadConversation(conversationId) {
        try {
            const response = await fetch(`/api/conversations/${conversationId}`);
            const data = await response.json();
            
            // Clear existing messages
            this.chatMessages.innerHTML = '';
            
            // Add messages
            data.messages.forEach(message => {
                this.addMessage(message);
                
                // Show intent analysis for customer messages
                if (message.sender_type === 'customer' && message.detected_intents) {
                    this.showIntentAnalysis(message.detected_intents, message.id);
                }
            });
            
            // Update conversation status UI
            this.updateConversationUI(data.conversation.status);
            
        } catch (error) {
            console.error('Failed to load conversation:', error);
            this.showNotification('Failed to load conversation', 'danger');
        }
    }
    
    sendMessage() {
        const content = this.messageInput.value.trim();
        if (!content || !this.currentConversationId) return;
        
        this.socket.emit('send_message', {
            conversation_id: this.currentConversationId,
            sender_type: 'agent',
            sender_name: this.agentName,
            content: content
        });
        
        this.messageInput.value = '';
    }
    
    addMessage(messageData) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${messageData.sender_type}`;
        messageDiv.dataset.messageId = messageData.message_id || messageData.id;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.textContent = messageData.content;
        
        const infoDiv = document.createElement('div');
        infoDiv.className = 'message-info';
        
        const time = new Date(messageData.timestamp).toLocaleTimeString();
        infoDiv.textContent = `${messageData.sender_name} • ${time}`;
        
        messageDiv.appendChild(bubbleDiv);
        messageDiv.appendChild(infoDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showIntentAnalysis(intentData, messageId) {
        if (!intentData || !intentData.intents) return;
        
        this.intentPanel.style.display = 'block';
        
        const analysisDiv = document.createElement('div');
        analysisDiv.className = 'intent-analysis mb-3 p-2 border rounded';
        analysisDiv.dataset.messageId = messageId;
        
        let html = '<div class="d-flex justify-content-between align-items-start mb-2">';
        html += `<strong>Message #${messageId}</strong>`;
        html += '</div>';
        
        // Primary intent and metadata
        html += '<div class="mb-2">';
        if (intentData.primary_intent) {
            html += `<span class="badge bg-primary me-2">${intentData.primary_intent}</span>`;
        }
        if (intentData.sentiment) {
            const sentimentColor = {
                'positive': 'success',
                'neutral': 'secondary',  
                'negative': 'danger'
            }[intentData.sentiment] || 'secondary';
            html += `<span class="badge bg-${sentimentColor} me-2">${intentData.sentiment}</span>`;
        }
        if (intentData.urgency) {
            const urgencyColor = {
                'high': 'danger',
                'medium': 'warning',
                'low': 'success'
            }[intentData.urgency] || 'secondary';
            html += `<span class="badge bg-${urgencyColor}">${intentData.urgency} priority</span>`;
        }
        html += '</div>';
        
        // All detected intents
        html += '<div class="intent-list">';
        intentData.intents.forEach(intent => {
            const confidence = Math.round(intent.confidence * 100);
            const confidenceClass = confidence >= 80 ? 'confidence-high' : 
                                  confidence >= 60 ? 'confidence-medium' : 'confidence-low';
            
            html += `
                <div class="intent-item mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="intent-name">${intent.intent}</span>
                        <small class="text-muted">${confidence}%</small>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
                    </div>
                    ${intent.reasoning ? `<small class="text-muted">${intent.reasoning}</small>` : ''}
                </div>
            `;
        });
        html += '</div>';
        
        // Annotation controls
        html += `
            <div class="mt-2">
                <button class="btn btn-sm btn-outline-primary annotate-btn" data-message-id="${messageId}">
                    <i data-feather="edit-3"></i> Annotate
                </button>
            </div>
        `;
        
        analysisDiv.innerHTML = html;
        
        // Add click handler for annotation
        analysisDiv.querySelector('.annotate-btn').addEventListener('click', () => {
            this.showAnnotationModal(messageId, intentData);
        });
        
        this.intentResults.appendChild(analysisDiv);
        feather.replace();
    }
    
    showAnnotationModal(messageId, intentData) {
        // Simple prompt-based annotation (could be enhanced with a proper modal)
        const correctIntent = prompt(
            `Current primary intent: ${intentData.primary_intent}\n\n` +
            'Enter the correct intent (or press Cancel to keep current):'
        );
        
        if (correctIntent && correctIntent.trim()) {
            const confidence = prompt('Confidence (0.0 - 1.0):', '1.0');
            const notes = prompt('Notes (optional):');
            
            this.socket.emit('annotate_intent', {
                message_id: messageId,
                agent_name: this.agentName,
                intent: correctIntent.trim(),
                confidence: parseFloat(confidence) || 1.0,
                notes: notes || ''
            });
            
            this.showNotification('Intent annotation saved', 'success');
        }
    }
    
    updateConversationStatus(status) {
        if (!this.currentConversationId) return;
        
        this.socket.emit('update_conversation_status', {
            conversation_id: this.currentConversationId,
            status: status,
            agent_name: this.agentName
        });
    }
    
    updateConversationUI(status) {
        if (status === 'closed') {
            this.messageInput.disabled = true;
            this.sendButton.disabled = true;
            this.closeButton.style.display = 'none';
            this.reopenButton.style.display = 'inline-block';
        } else {
            this.messageInput.disabled = false;
            this.sendButton.disabled = false;
            this.closeButton.style.display = 'inline-block';
            this.reopenButton.style.display = 'none';
        }
    }
    
    addNewConversationToList(conversationData) {
        const conversationDiv = document.createElement('div');
        conversationDiv.className = 'conversation-item p-3 border-bottom';
        conversationDiv.dataset.conversationId = conversationData.conversation_id;
        conversationDiv.dataset.customerName = conversationData.customer_name;
        conversationDiv.dataset.subject = conversationData.subject;
        
        conversationDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <h6 class="mb-1">${conversationData.customer_name}</h6>
                    <p class="mb-1 small text-muted">${conversationData.subject}</p>
                    <small class="text-muted">Just now</small>
                </div>
                <span class="badge bg-success">active</span>
            </div>
        `;
        
        // Add to top of list
        this.conversationsList.insertBefore(conversationDiv, this.conversationsList.firstChild);
        
        // Update conversation count
        const countBadge = document.getElementById('conversation-count');
        const currentCount = parseInt(countBadge.textContent) || 0;
        countBadge.textContent = currentCount + 1;
    }
    
    updateConversationInList(conversationId) {
        const conversationItem = this.conversationsList.querySelector(
            `.conversation-item[data-conversation-id="${conversationId}"]`
        );
        
        if (conversationItem) {
            // Move to top and update timestamp
            const timeElement = conversationItem.querySelector('small.text-muted');
            if (timeElement) {
                timeElement.textContent = 'Just now';
            }
            this.conversationsList.insertBefore(conversationItem, this.conversationsList.firstChild);
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showNotification(message, type = 'info') {
        // Create notification toast
        const toastDiv = document.createElement('div');
        toastDiv.className = `toast align-items-center text-bg-${type} border-0`;
        toastDiv.setAttribute('role', 'alert');
        toastDiv.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // Add to page
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.appendChild(toastDiv);
        
        // Show toast
        const toast = new bootstrap.Toast(toastDiv);
        toast.show();
        
        // Remove from DOM after hidden
        toastDiv.addEventListener('hidden.bs.toast', () => {
            toastDiv.remove();
        });
    }
}

// Initialize agent dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AgentDashboard();
    
    // Refresh feather icons after dynamic content
    const observer = new MutationObserver(() => {
        feather.replace();
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
