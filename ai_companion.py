from typing import Dict, List
import random
from datetime import datetime
from transformers import pipeline
import logging

class AICompanion:
    def __init__(self, memory_handler):
        self.memory_handler = memory_handler
        
        # Initialize a smaller language model for conversation
        self.language_model = pipeline("text-generation", model="distilgpt2")  # Use a smaller model

        
        # Enhanced reflection prompts based on emotions
        self.reflection_prompts = {
            "joy": [
                "What made this moment particularly special for you?",
                "How can you create more joyful moments like this in your life?",
                "Who were the people that contributed to this happiness?",
                "What does this happy memory teach you about what brings you joy?"
            ],
            "sadness": [
                "What helped you cope with these feelings?",
                "What have you learned from this experience?",
                "How have you grown since this moment?",
                "What support systems were most helpful during this time?"
            ],
            "anger": [
                "What triggered these feelings?",
                "How did you manage your response?",
                "What could be a more constructive way to handle similar situations?",
                "What does this memory teach you about your boundaries?"
            ],
            "anxiety": [
                "What coping strategies worked best for you?",
                "What helped you feel more grounded in this moment?",
                "How can you better prepare for similar situations?",
                "What resources or support could help you next time?"
            ],
            "gratitude": [
                "What makes this memory particularly meaningful?",
                "How has this experience shaped your perspective?",
                "Who would you like to thank for this moment?",
                "How can you cultivate more moments like this?"
            ],
            "love": [
                "What makes this connection special to you?",
                "How has this relationship helped you grow?",
                "What values does this memory reflect?",
                "How can you nurture more connections like this?"
            ]
        }
        
        # Add general reflection prompts
        self.general_prompts = [
            "Would you like to explore any similar memories?",
            "How do you feel reflecting on this memory now?",
            "What patterns do you notice in your emotional responses?",
            "What would you tell your past self about this experience?"
        ]

    def _analyze_user_emotion(self, user_input: str) -> str:
        """Analyze user input to detect emotional context"""
        # Use emotion analyzer from memory handler
        analysis = self.memory_handler.emotion_analyzer.analyze_text(user_input)
        if analysis["emotion_tags"]:
            return analysis["emotion_tags"][0]
        return "neutral"

    def _get_relevant_prompt(self, emotion: str) -> str:
        """Get a relevant reflection prompt based on emotion"""
        if emotion in self.reflection_prompts:
            return random.choice(self.reflection_prompts[emotion])
        return random.choice(self.general_prompts)

    async def chat(self, user_input: str) -> Dict:
        try:
            # Analyze user's emotional state
            user_emotion = self._analyze_user_emotion(user_input)
            
            # Prepare response
            response = {
                "message": "",
                "related_memories": [],
                "reflection_prompt": None,
                "detected_emotion": user_emotion
            }
            
            # Retrieve relevant memories, handle case if no memories exist
            try:
                memories = await self.memory_handler.retrieve_memories(
                    query=user_input,
                    emotion=user_emotion,
                    limit=3
                )
            except Exception as e:
                logging.error(f"Error retrieving memories: {str(e)}")
                memories = []  # Set memories to an empty list if retrieval fails

            
            # Generate conversational response
            if memories:
                response["message"] = self._generate_memory_response(
                    user_emotion, memories
                )
                response["related_memories"] = memories
            else:
                response["message"] = self._generate_exploratory_response(
                    user_emotion
                )
            
            # Add reflection prompt
            response["reflection_prompt"] = self._get_relevant_prompt(user_emotion)
            
            return response
            
        except Exception as e:
            raise Exception(f"Error in AI companion chat: {str(e)}")

    def _generate_memory_response(self, emotion: str, memories: List[Dict]) -> str:
        """Generate a response that incorporates retrieved memories"""
        memory_count = len(memories)
        recent_memory = memories[0]
        
        responses = [
            f"I notice you're feeling {emotion}. This reminds me of a memory you shared about {recent_memory['caption']}.",
            f"Your feeling of {emotion} connects with {memory_count} memories you've shared with me.",
            f"As we discuss this, I'm reminded of your memory about {recent_memory['caption']}. Would you like to reflect on it?",
            f"I see a connection between your current feelings and your memory of {recent_memory['caption']}."
        ]
        
        return random.choice(responses)

    def _generate_exploratory_response(self, emotion: str) -> str:
        """Generate a response when no specific memories are found"""
        responses = [
            f"I notice you're feeling {emotion}. Would you like to share more about what's on your mind?",
            f"It sounds like you're experiencing {emotion}. I'm here to listen and reflect with you.",
            f"Thank you for sharing these {emotion} feelings. Would you like to explore them together?",
            f"I'm here to support you as you process these {emotion} emotions. What would be most helpful to discuss?"
        ]
        
        return random.choice(responses)
