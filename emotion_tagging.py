from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class EmotionAnalyzer:
    def __init__(self):
        # Initialize emotion classification model
        self.emotion_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-uncased-emotion",
            return_all_scores=True
        )
        
        # Initialize text embedding model for semantic search
        self.text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name)
        
        # Initialize CLIP model for image-text alignment
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Define emotion categories
        self.emotion_categories = [
            "joy", "sadness", "anger", "fear", "love", 
            "surprise", "neutral", "anxiety", "gratitude"
        ]

    def analyze_text(self, text: str) -> Dict:
        # Get detailed emotion analysis
        emotion_scores = self.emotion_classifier(text)[0]
        
        # Generate text embeddings for semantic search
        inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embedding = self.text_model(**inputs).logits[0].numpy()
        
        # Get primary emotions (those with score > threshold)
        threshold = 0.2
        primary_emotions = [
            score["label"] for score in emotion_scores 
            if score["score"] > threshold
        ]
        
        return {
            "emotion_tags": primary_emotions,
            "emotion_scores": {
                score["label"]: score["score"] 
                for score in emotion_scores
            },
            "text_embedding": text_embedding.tolist()
        }

    def analyze_image(self, image_path: str) -> Dict:
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Generate image embeddings
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Prepare emotion-related text prompts
            emotion_prompts = [
                f"an image expressing {emotion}"
                for emotion in self.emotion_categories
            ]
            
            text_inputs = self.clip_processor(
                text=emotion_prompts,
                return_tensors="pt",
                padding=True
            )
            
            # Get text features for emotion prompts
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
            
            # Calculate similarity scores
            similarity = torch.nn.functional.cosine_similarity(
                image_features, text_features
            )
            
            # Get emotion scores
            emotion_scores = {
                emotion: score.item()
                for emotion, score in zip(self.emotion_categories, similarity[0])
            }
            
            # Get primary emotions (those with score > threshold)
            threshold = 0.2
            primary_emotions = [
                emotion for emotion, score in emotion_scores.items()
                if score > threshold
            ]
            
            return {
                "image_embedding": image_features[0].tolist(),
                "emotion_scores": emotion_scores,
                "primary_emotions": primary_emotions
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")

    def combine_analysis(self, text_analysis: Dict, image_analysis: Dict) -> Dict:
        """Combine text and image analysis for overall emotional context"""
        # Combine emotion scores with weights
        text_weight = 0.6
        image_weight = 0.4
        
        combined_scores = {}
        for emotion in self.emotion_categories:
            text_score = text_analysis["emotion_scores"].get(emotion, 0)
            image_score = image_analysis["emotion_scores"].get(emotion, 0)
            combined_scores[emotion] = (
                text_score * text_weight + image_score * image_weight
            )
        
        # Get primary emotions from combined scores
        threshold = 0.2
        primary_emotions = [
            emotion for emotion, score in combined_scores.items()
            if score > threshold
        ]
        
        return {
            "primary_emotions": primary_emotions,
            "emotion_scores": combined_scores,
            "text_analysis": text_analysis,
            "image_analysis": image_analysis
        }