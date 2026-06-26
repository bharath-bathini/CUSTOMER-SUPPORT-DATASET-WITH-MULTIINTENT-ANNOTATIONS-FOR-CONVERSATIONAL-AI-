import os
import logging
import json
import requests
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Intent(BaseModel):
    intent: str
    confidence: float
    reasoning: str

class IntentClassification(BaseModel):
    intents: List[Intent]

class IntentClassifier:
    def __init__(self):
        """Initialize intent classifier using Gemini API"""
        self.api_key = "AIzaSyBpLetJX8VwYq3QHc_icj9po2DWiQRqMdY"
        self.client_available = bool(self.api_key)
        if self.client_available:
            logger.info("Gemini AI intent classifier initialized")
        else:
            logger.error("API key missing; AI service unavailable")

    def classify_intents(self, message):
        """Classify the message into a meaningful category dynamically"""
        if not message or not message.strip():
            return [{"intent": "general", "confidence": 0.5, "reasoning": "Empty input classified as general"}]

        if self.client_available:
            return self._classify_with_ai(message)
        else:
            return self._fallback(message)

    def _classify_with_ai(self, message):
        system_prompt = """
You are an expert topic classifier. Analyze the user's message or phrase and determine its main category/topic dynamically. 
Guidelines:
1. Identify the primary topic/domain of the input (e.g., medical, education, finance, travel, technology, legal, etc.)
2. Assign a confidence score from 0.0 to 1.0
3. Include a brief reasoning
4. Respond ONLY in JSON in this format:
{"intents": [{"intent": "category_name", "confidence": 0.85, "reasoning": "brief explanation"}]}
"""

        user_prompt = f"Classify this input into a single-word category as per domain: '{message}'"

        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "config": {"systemInstruction": system_prompt, "temperature": 0.3, "maxOutputTokens": 200}
        }

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            text_output = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if text_output:
                try:
                    result = json.loads(text_output)
                    intents = result.get("intents", [])
                    validated = []
                    for i in intents:
                        if "intent" in i and "confidence" in i:
                            validated.append({
                                "intent": i["intent"],
                                "confidence": min(1.0, max(0.0, float(i["confidence"]))),
                                "reasoning": i.get("reasoning", "AI classification")
                            })
                    if validated:
                        return validated[:3]
                    else:
                        return self._fallback(message)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    return self._fallback(message)
            else:
                return self._fallback(message)
        except Exception as e:
            logger.error(f"Error in AI classification: {e}")
            return self._fallback(message)

    def _fallback(self, message):
        """Fallback if API fails or returns empty"""
        return [{"intent": "general", "confidence": 0.5, "reasoning": "Could not classify dynamically; defaulted to general"}]

    def is_available(self):
        return self.client_available
