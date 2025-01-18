from typing import Dict, List
import random
from datetime import datetime, timedelta

class AICompanion:
    def __init__(self, memory_handler):
        self.memory_handler = memory_handler
        self.reflection_prompts = {
            "happy": [
                "What made this moment particularly joyful?",
                "How can you create more moments like this?",
                "Who were the people that contributed to this happiness?"
            ],
            "sad": [
                "What helped you get through this difficult time?",
                "What have you learned from this experience?",
                "How have you grown since this moment?"
            ],
            "stressed": [
                "What coping strategies worked best for you?",
                "What support systems did you rely on?",
                "How can you better prepare for similar situations?"
            ]
        }

    async def chat(self, user_input: str, user_emotion: str = None) -> Dict:
        # Analyze user input for emotional content
        response = {
            "message": "",
            "related_memories": [],
            "reflection_prompt": None
        }
        
        # Retrieve relevant memories based on input
        memories = await self.memory_handler.retrieve_memories(query=user_input)
        
        if "stress" in user_input.lower():
            response["message"] = "I notice you're talking about stress. Let's explore some past experiences that might help."
            response["reflection_prompt"] = random.choice(self.reflection_prompts["stressed"])
        
        elif "happy" in user_input.lower() or "joy" in user_input.lower():
            response["message"] = "It's wonderful to focus on positive moments. Let's explore what makes you happy."
            response["reflection_prompt"] = random.choice(self.reflection_prompts["happy"])
        
        elif "sad" in user_input.lower() or "down" in user_input.lower():
            response["message"] = "I hear that you're feeling down. Let's reflect on how you've handled similar feelings before."
            response["reflection_prompt"] = random.choice(self.reflection_prompts["sad"])
        
        else:
            response["message"] = "I'm here to help you reflect on your experiences. Would you like to explore specific emotions or memories?"
        
        # Add relevant memories to response
        if memories:
            response["related_memories"] = memories[:3]  # Limit to 3 most relevant memories
            response["message"] += "\n\nI found some related memories that might be worth reflecting on."
        
        return response