from transformers import pipeline
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class EmotionAnalyzer:
    def __init__(self):
        # Initialize emotion classification model
        self.emotion_classifier = pipeline("text-classification", 
                                        model="bhadresh-savani/bert-base-uncased-emotion")
        
        # Initialize text embedding model
        self.text_embedding_model = pipeline("feature-extraction", 
                                           model="distilbert-base-uncased")
        
        # Initialize CLIP model for image analysis
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def analyze_text(self, text: str) -> Dict:
        # Get emotion classifications
        emotions = self.emotion_classifier(text)
        
        # Generate text embeddings
        embeddings = self.text_embedding_model(text)
        
        return {
            "emotion_tags": [emotion["label"] for emotion in emotions],
            "emotion_scores": [emotion["score"] for emotion in emotions],
            "text_embedding": embeddings[0][0].tolist()  # First token embedding
        }

    def analyze_image(self, image_path: str) -> Dict:
        # Load and process image
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # Generate image embeddings
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        # Pre-defined emotion concepts for CLIP
        emotion_concepts = [
            "a happy moment", "a sad moment", "an exciting moment",
            "a peaceful moment", "an anxious moment", "a loving moment"
        ]
        
        # Get emotion scores using CLIP
        text_inputs = self.clip_processor(text=emotion_concepts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            
        # Calculate similarity scores
        similarity = torch.nn.functional.cosine_similarity(
            image_features, text_features
        )
        
        return {
            "image_embedding": image_features[0].tolist(),
            "emotion_scores": {
                concept: score.item()
                for concept, score in zip(emotion_concepts, similarity[0])
            }
        }