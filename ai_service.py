import os
import logging
import json
import base64
import requests

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        """Initialize AI service with direct Gemini API requests"""
        # Hardcoded API key (⚠️ best practice: store in env variable)
        self.api_key = "AIzaSyBpLetJX8VwYq3QHc_icj9po2DWiQRqMdY"
        if not self.api_key:
            logger.error("API key is missing")
            self.client_available = False
        else:
            self.client_available = True
            logger.info("Google Gemini AI direct API client initialized")

    def generate_response(self, user_message, conversation_history=None, detected_intents=None, image_path=None):
        """Generate AI response using Gemini API (supports text + optional image)"""
        if not self.client_available:
            return self._fallback_response(user_message, detected_intents)

        try:
            # Build conversation context
            context = ""
            if conversation_history:
                recent_messages = conversation_history[-6:]
                for msg in recent_messages:
                    role = "Customer" if msg['role'] == 'user' else "Support Agent"
                    context += f"{role}: {msg['content']}\n"

            # Build intent context
            intent_context = ""
            if detected_intents:
                intent_list = [f"- {intent['intent']} (confidence: {intent['confidence']:.2f})"
                               for intent in detected_intents]
                intent_context = f"\n\nDetected customer intents:\n" + "\n".join(intent_list)

            # Prepare system + user prompts
            system_prompt = """You are a helpful Gemini AI"""

            user_prompt = f""" {user_message} 
            Please provide a detailed response."""

            # API URL
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.api_key}"

            # Prepare request payload
            parts = [{"text": user_prompt}]
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")
                parts.insert(0, {"inlineData": {"mimeType": "image/jpeg", "data": image_data}})

            payload = {"contents": [{"role": "user", "parts": parts}]}

            # Send request
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            text_output = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            if text_output:
                return text_output.strip()
            else:
                logger.warning("Empty response from Gemini API")
                return self._fallback_response(user_message, detected_intents)

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return self._fallback_response(user_message, detected_intents)

    def _fallback_response(self, user_message, detected_intents=None):
        """Fallback if Gemini API is unavailable"""
        if detected_intents:
            primary_intent = detected_intents[0]['intent'].lower()
            if 'technical' in primary_intent:
                return "I understand you're experiencing a technical issue. Could you provide more details so we can troubleshoot?"
            elif 'billing' in primary_intent:
                return "I can help you with billing. Could you please share your account details or describe the billing concern?"
            elif 'account' in primary_intent:
                return "I can help with your account. What specific issue are you facing?"
            elif 'product' in primary_intent:
                return "I'd be happy to provide product details. Which product or service are you asking about?"
            elif 'general' in primary_intent:
                return "Thanks for contacting us! Could you tell me more about how I can assist you?"
        return "Thank you for reaching out! Could you provide more details so I can assist you better?"

    def is_available(self):
        return self.client_available