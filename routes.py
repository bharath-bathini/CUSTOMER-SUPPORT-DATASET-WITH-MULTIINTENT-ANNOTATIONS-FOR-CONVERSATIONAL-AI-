from flask import render_template, request, jsonify, redirect, url_for
from flask_socketio import emit, join_room, leave_room
from app import app, socketio, db
from models import Conversation, Message, IntentAnnotation
from intent_detector import detect_intents, get_conversation_context
import logging

logger = logging.getLogger(__name__)

# Store active connections
active_connections = {}

@app.route('/')
def index():
    """Landing page for the customer support system"""
    return render_template('index.html')

@app.route('/customer/chat')
def customer_chat():
    """Customer chat interface"""
    return render_template('customer_chat.html')

@app.route('/agent/dashboard')
def agent_dashboard():
    """Agent dashboard for managing conversations"""
    conversations = Conversation.query.order_by(Conversation.updated_at.desc()).all()
    return render_template('agent_dashboard.html', conversations=conversations)

@app.route('/api/conversations/<int:conversation_id>')
def get_conversation(conversation_id):
    """Get conversation details with messages"""
    conversation = Conversation.query.get_or_404(conversation_id)
    messages = Message.query.filter_by(conversation_id=conversation_id)\
                           .order_by(Message.timestamp.asc()).all()
    
    return jsonify({
        'conversation': {
            'id': conversation.id,
            'customer_name': conversation.customer_name,
            'customer_email': conversation.customer_email,
            'subject': conversation.subject,
            'status': conversation.status,
            'created_at': conversation.created_at.isoformat(),
            'updated_at': conversation.updated_at.isoformat()
        },
        'messages': [{
            'id': msg.id,
            'sender_type': msg.sender_type,
            'sender_name': msg.sender_name,
            'content': msg.content,
            'timestamp': msg.timestamp.isoformat(),
            'detected_intents': msg.detected_intents,
            'intent_processed': msg.intent_processed
        } for msg in messages]
    })

# SocketIO Events
@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'msg': 'Connected to ConvoSense'})

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    # Remove from active connections
    if request.sid in active_connections:
        del active_connections[request.sid]

@socketio.on('start_conversation')
def handle_start_conversation(data):
    """Start a new conversation"""
    try:
        # Create new conversation
        conversation = Conversation(
            customer_name=data['customer_name'],
            customer_email=data['customer_email'],
            subject=data['subject']
        )
        db.session.add(conversation)
        db.session.commit()
        
        # Store connection info
        active_connections[request.sid] = {
            'conversation_id': conversation.id,
            'user_type': 'customer',
            'user_name': data['customer_name']
        }
        
        # Join the conversation room
        join_room(f"conversation_{conversation.id}")
        
        # Send initial system message
        system_message = Message(
            conversation_id=conversation.id,
            sender_type='system',
            sender_name='ConvoSense',
            content=f"Conversation started. Customer: {data['customer_name']} ({data['customer_email']})"
        )
        db.session.add(system_message)
        db.session.commit()
        
        emit('conversation_started', {
            'conversation_id': conversation.id,
            'message': 'Connected! How can we help you today?'
        })
        
        # Notify agents of new conversation
        socketio.emit('new_conversation', {
            'conversation_id': conversation.id,
            'customer_name': data['customer_name'],
            'subject': data['subject']
        }, room='agents')
        
        logger.info(f"New conversation started: {conversation.id}")
        
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        emit('error', {'message': 'Failed to start conversation'})

@socketio.on('join_agent_room')
def handle_join_agent_room(data):
    """Agent joins the agent room for notifications"""
    join_room('agents')
    active_connections[request.sid] = {
        'user_type': 'agent',
        'user_name': data.get('agent_name', 'Agent')
    }
    emit('joined_agents', {'status': 'success'})

@socketio.on('join_conversation')
def handle_join_conversation(data):
    """Agent joins a specific conversation"""
    conversation_id = data['conversation_id']
    agent_name = data.get('agent_name', 'Agent')
    
    join_room(f"conversation_{conversation_id}")
    
    # Update connection info
    if request.sid in active_connections:
        active_connections[request.sid]['conversation_id'] = conversation_id
        active_connections[request.sid]['agent_name'] = agent_name
    
    # Send agent joined message
    agent_message = Message(
        conversation_id=conversation_id,
        sender_type='system',
        sender_name='ConvoSense',
        content=f"{agent_name} joined the conversation"
    )
    db.session.add(agent_message)
    db.session.commit()
    
    emit('agent_joined', {
        'agent_name': agent_name,
        'message': f"{agent_name} has joined the conversation"
    }, room=f"conversation_{conversation_id}")

@socketio.on('send_message')
def handle_send_message(data):
    """Handle incoming messages from customers or agents"""
    try:
        conversation_id = data['conversation_id']
        sender_type = data['sender_type']  # 'customer' or 'agent'
        sender_name = data['sender_name']
        content = data['content']
        
        # Create message record
        message = Message(
            conversation_id=conversation_id,
            sender_type=sender_type,
            sender_name=sender_name,
            content=content
        )
        db.session.add(message)
        db.session.commit()
        
        # Prepare message data for broadcast
        message_data = {
            'message_id': message.id,
            'conversation_id': conversation_id,
            'sender_type': sender_type,
            'sender_name': sender_name,
            'content': content,
            'timestamp': message.timestamp.isoformat()
        }
        
        # If it's a customer message, detect intents
        if sender_type == 'customer':
            try:
                # Get conversation context
                context = get_conversation_context(conversation_id)
                
                # Detect intents
                intent_result = detect_intents(content, context)
                
                # Update message with detected intents
                message.detected_intents = intent_result
                message.intent_processed = True
                db.session.commit()
                
                # Add intent data to message
                message_data['detected_intents'] = intent_result
                message_data['intent_processed'] = True
                
            except Exception as e:
                logger.error(f"Intent detection failed: {e}")
                message_data['detected_intents'] = None
                message_data['intent_processed'] = False
        
        # Broadcast message to all participants in the conversation
        socketio.emit('new_message', message_data, room=f"conversation_{conversation_id}")
        
        # Update conversation timestamp
        conversation = Conversation.query.get(conversation_id)
        if conversation:
            conversation.updated_at = message.timestamp
            db.session.commit()
        
        logger.info(f"Message sent in conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        emit('error', {'message': 'Failed to send message'})

@socketio.on('annotate_intent')
def handle_annotate_intent(data):
    """Handle intent annotation by agents"""
    try:
        message_id = data['message_id']
        agent_name = data['agent_name']
        annotated_intent = data['intent']
        confidence = data.get('confidence', 1.0)
        notes = data.get('notes', '')
        
        # Create annotation record
        annotation = IntentAnnotation(
            message_id=message_id,
            agent_name=agent_name,
            annotated_intent=annotated_intent,
            confidence=confidence,
            notes=notes
        )
        db.session.add(annotation)
        db.session.commit()
        
        emit('intent_annotated', {
            'message_id': message_id,
            'annotation': {
                'intent': annotated_intent,
                'confidence': confidence,
                'agent_name': agent_name,
                'notes': notes
            }
        })
        
        logger.info(f"Intent annotated for message {message_id}")
        
    except Exception as e:
        logger.error(f"Error annotating intent: {e}")
        emit('error', {'message': 'Failed to annotate intent'})

@socketio.on('update_conversation_status')
def handle_update_conversation_status(data):
    """Update conversation status (close, reopen, etc.)"""
    try:
        conversation_id = data['conversation_id']
        new_status = data['status']
        agent_name = data.get('agent_name', 'Agent')
        
        conversation = Conversation.query.get(conversation_id)
        if conversation:
            conversation.status = new_status
            db.session.commit()
            
            # Send status update message
            status_message = Message(
                conversation_id=conversation_id,
                sender_type='system',
                sender_name='ConvoSense',
                content=f"Conversation status updated to: {new_status} by {agent_name}"
            )
            db.session.add(status_message)
            db.session.commit()
            
            socketio.emit('conversation_status_updated', {
                'conversation_id': conversation_id,
                'status': new_status,
                'agent_name': agent_name
            }, room=f"conversation_{conversation_id}")
            
            logger.info(f"Conversation {conversation_id} status updated to {new_status}")
        
    except Exception as e:
        logger.error(f"Error updating conversation status: {e}")
        emit('error', {'message': 'Failed to update conversation status'})
