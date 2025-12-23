"""
AI Chatbot Service using Groq API
Provides health assistance, symptom checking, and health tips
"""

import os
import json
import requests
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthChatbot:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Health knowledge base
        self.health_tips = {
            "diabetes": [
                "Monitor your blood sugar levels regularly",
                "Maintain a balanced diet with low glycemic index foods",
                "Exercise for at least 150 minutes per week",
                "Stay hydrated and limit sugary drinks",
                "Get regular check-ups with your healthcare provider"
            ],
            "heart_disease": [
                "Maintain a heart-healthy diet rich in fruits and vegetables",
                "Exercise regularly - aim for 30 minutes daily",
                "Quit smoking and avoid secondhand smoke",
                "Manage stress through relaxation techniques",
                "Monitor your blood pressure and cholesterol levels"
            ],
            "cancer": [
                "Maintain a healthy weight and active lifestyle",
                "Eat a diet rich in fruits, vegetables, and whole grains",
                "Limit processed and red meats",
                "Avoid tobacco and limit alcohol consumption",
                "Get regular screenings as recommended by your doctor"
            ]
        }
        
        self.symptom_keywords = {
            "diabetes": ["thirst", "urination", "fatigue", "blurred vision", "weight loss", "hunger"],
            "heart_disease": ["chest pain", "shortness of breath", "fatigue", "palpitations", "dizziness", "swelling"],
            "cancer": ["lump", "unusual bleeding", "weight loss", "fatigue", "pain", "changes in skin"]
        }

    def get_health_tip(self, condition: str) -> str:
        """Get personalized health tips based on condition"""
        tips = self.health_tips.get(condition.lower(), [])
        if tips:
            import random
            return random.choice(tips)
        return "Maintain a healthy lifestyle with regular exercise and balanced nutrition."

    def analyze_symptoms(self, symptoms: str) -> Dict:
        """Analyze symptoms and provide preliminary assessment"""
        symptoms_lower = symptoms.lower()
        matched_conditions = []
        
        for condition, keywords in self.symptom_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in symptoms_lower)
            if matches > 0:
                matched_conditions.append({
                    'condition': condition,
                    'match_count': matches,
                    'confidence': min(matches / len(keywords) * 100, 100)
                })
        
        # Sort by confidence
        matched_conditions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'matched_conditions': matched_conditions[:3],
            'recommendation': self._get_symptom_recommendation(matched_conditions)
        }

    def _get_symptom_recommendation(self, conditions: List[Dict]) -> str:
        """Get recommendation based on symptom analysis"""
        if not conditions:
            return "Your symptoms don't match common patterns. Consider consulting a healthcare professional for proper evaluation."
        
        top_condition = conditions[0]
        if top_condition['confidence'] > 50:
            return f"Based on your symptoms, there may be a risk of {top_condition['condition'].replace('_', ' ')}. Please consult a healthcare professional for proper diagnosis."
        else:
            return "Your symptoms show some patterns but require professional evaluation. Please schedule an appointment with your doctor."

    def chat_with_groq(self, message: str, context: str = "") -> str:
        """Send message to Groq API and get response"""
        if not self.api_key:
            return "AI assistant is currently unavailable. Please try again later."
        
        try:
            # Prepare the prompt with health context
            system_prompt = f"""You are a helpful health assistant. Provide accurate, helpful, and safe health information.
            
            Important guidelines:
            - Always recommend consulting healthcare professionals for serious concerns
            - Provide general health tips and information
            - Be encouraging and supportive
            - Never provide specific medical diagnoses
            - Focus on prevention and healthy lifestyle choices
            
            Context: {context}
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {str(e)}")
            return "I'm having trouble connecting to the AI service. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected error in chat: {str(e)}")
            return "An unexpected error occurred. Please try again."

    def get_quick_health_tips(self) -> List[Dict]:
        """Get quick health tips for the homepage"""
        return [
            {
                "icon": "💧",
                "title": "Stay Hydrated",
                "description": "Drink at least 8 glasses of water daily for optimal health"
            },
            {
                "icon": "🏃‍♂️",
                "title": "Regular Exercise",
                "description": "Aim for 150 minutes of moderate exercise per week"
            },
            {
                "icon": "🥗",
                "title": "Balanced Diet",
                "description": "Include fruits, vegetables, and whole grains in your meals"
            },
            {
                "icon": "😴",
                "title": "Quality Sleep",
                "description": "Get 7-9 hours of sleep for proper rest and recovery"
            },
            {
                "icon": "🧘‍♀️",
                "title": "Manage Stress",
                "description": "Practice relaxation techniques like meditation or yoga"
            },
            {
                "icon": "🏥",
                "title": "Regular Check-ups",
                "description": "Schedule annual health screenings and check-ups"
            }
        ]

    def get_emergency_contacts(self) -> List[Dict]:
        """Get emergency contact information"""
        return [
            {
                "name": "Emergency Services",
                "number": "108",
                "description": "Call for immediate medical emergencies"
            },
            {
                "name": "Poison Control",
                "number": "1800-425-1890",
                "description": "24/7 poison emergency hotline"
            },
            {
                "name": "Crisis Text Line",
                "number": "Text HOME to 741741",
                "description": "24/7 crisis support via text"
            }
        ]

# Global chatbot instance
chatbot = HealthChatbot()
