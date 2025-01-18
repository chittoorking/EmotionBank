from sqlalchemy import Column, Integer, String, Text, JSON, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    caption = Column(String, index=True)
    content = Column(Text)
    emotional_tags = Column(JSON)  # List of emotion tags
    suggested_tags = Column(JSON)  # AI-suggested emotion tags
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String)  # Path to stored image
    text_embedding = Column(JSON)  # BERT embeddings for text
    image_embedding = Column(JSON)  # CLIP embeddings for image
    sentiment_scores = Column(JSON)  # Detailed emotion analysis scores

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_input = Column(Text)
    ai_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    related_memory_ids = Column(JSON)  # IDs of memories referenced in chat