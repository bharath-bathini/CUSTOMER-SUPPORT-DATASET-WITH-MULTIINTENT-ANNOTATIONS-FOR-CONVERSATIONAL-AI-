import json
import os
import logging
import requests

# Using Google Gemini 1.5 Flash for intent detection
GEMINI_API_KEY = "AIzaSyBpLetJX8VwYq3QHc_icj9po2DWiQRqMdY"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

logger = logging.getLogger(__name__)

def detect_intents(message_content, conversation_context=None):
    """
    Detect multiple intents in a customer support message using Google Gemini 1.5 Flash
    Returns a list of intents with confidence scores
    """
    try:
        # Define the system prompt for multi-intent detection
        system_instruction = """You are an AI assistant specialized in customer support intent detection. 
        Analyze the given message and identify ALL possible intents present in the customer's message.
        
        Common customer support intents include:
        - billing_inquiry
        - technical_support
        - product_information
        - complaint
        - refund_request
        - account_assistance
        - shipping_inquiry
        - feature_request
        - cancellation
        - general_inquiry
        - praise_feedback
        - bug_report
        
        For each detected intent, provide a confidence score between 0.0 and 1.0.
        Only include intents with confidence >= 0.3.
        
        Respond with JSON in this format:
        {
            "intents": [
                {"intent": "intent_name", "confidence": 0.85, "reasoning": "brief explanation"},
                {"intent": "another_intent", "confidence": 0.65, "reasoning": "brief explanation"}
            ],
            "primary_intent": "most_likely_intent",
            "sentiment": "positive|neutral|negative",
            "urgency": "low|medium|high"
        }"""
        
        # Prepare the user message with context if available
        user_message = f"Customer message: {message_content}"
        if conversation_context:
            user_message += f"\n\nConversation context: {conversation_context}"
        
        # Prepare the request payload for Gemini
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_instruction}\n\nAnalyze this customer message:\n{user_message}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "candidateCount": 1,
                "maxOutputTokens": 1000,
                "responseMimeType": "application/json"
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make the API request to Gemini
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Extract the generated content
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            result = json.loads(generated_text)
            logger.info(f"Intent detection successful: {result}")
            return result
        else:
            raise Exception("No candidates in Gemini response")
        
    except Exception as e:
        logger.error(f"Failed to detect intents: {e}")
        # Return fallback response
        return {
            "intents": [{"intent": "general_inquiry", "confidence": 0.5, "reasoning": "fallback due to API error"}],
            "primary_intent": "general_inquiry",
            "sentiment": "neutral",
            "urgency": "medium"
        }

def get_conversation_context(conversation_id, limit=5):
    """
    Get recent conversation context for better intent detection
    """
    from models import Message
    
    try:
        recent_messages = Message.query.filter_by(conversation_id=conversation_id)\
                                    .order_by(Message.timestamp.desc())\
                                    .limit(limit).all()
        
        context = []
        for msg in reversed(recent_messages):
            context.append(f"{msg.sender_type}: {msg.content[:200]}")
        
        return " | ".join(context)
    except Exception as e:
        logger.error(f"Failed to get conversation context: {e}")
        return None
