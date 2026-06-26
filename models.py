from app import db
from datetime import datetime
from sqlalchemy import Text, JSON

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_name = db.Column(db.String(100), nullable=False)
    customer_email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='active')  # active, closed, pending
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to messages
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sender_type = db.Column(db.String(20), nullable=False)  # customer, agent, system
    sender_name = db.Column(db.String(100), nullable=False)
    content = db.Column(Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # AI Intent Detection Results
    detected_intents = db.Column(JSON)  # Store multiple intents with confidence scores
    intent_processed = db.Column(db.Boolean, default=False)

class IntentAnnotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)
    agent_name = db.Column(db.String(100), nullable=False)
    annotated_intent = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, default=1.0)
    notes = db.Column(Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    message = db.relationship('Message', backref='annotations', lazy=True)
