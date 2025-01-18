import logging
from sqlalchemy.orm import Session
from fastapi import UploadFile
import os
import aiofiles
import numpy as np
from datetime import datetime
from typing import List, Dict
from models import Memory  # Added this import
import chromadb
from chromadb.config import Settings

class MemoryHandler:
    def __init__(self, db: Session, emotion_analyzer):
        self.db = db
        self.emotion_analyzer = emotion_analyzer
        
        # Initialize ChromaDB for vector search
        self.vector_db = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./vector_db"
        ))
        self.collection = self.vector_db.create_collection(name="memories")
        
        # Ensure directories exist
        os.makedirs("images", exist_ok=True)

    async def upload_memory(self, memory: Memory, file: UploadFile) -> Dict:
        try:
            # Save image file
            file_location = f"images/{file.filename}"
            async with aiofiles.open(file_location, "wb") as buffer:
                content = await file.read()
                await buffer.write(content)
            
            # Analyze text content
            text_analysis = self.emotion_analyzer.analyze_text(memory.content)
            
            # Analyze image
            image_analysis = self.emotion_analyzer.analyze_image(file_location)
            
            # Update memory object with analysis results
            memory.image_path = file_location
            memory.text_embedding = text_analysis["text_embedding"]
            memory.image_embedding = image_analysis["image_embedding"]
            memory.suggested_tags = text_analysis["emotion_tags"]
            memory.sentiment_scores = {
                "text": text_analysis["emotion_scores"],
                "image": image_analysis["emotion_scores"]
            }
            
            # Save to database
            self.db.add(memory)
            self.db.commit()
            self.db.refresh(memory)
            
            # Add to vector database for similarity search
            self.collection.add(
                ids=[str(memory.id)],
                embeddings=[memory.text_embedding],
                metadatas=[{
                    "caption": memory.caption,
                    "emotional_tags": memory.emotional_tags,
                    "timestamp": memory.timestamp.isoformat()
                }]
            )
            
            return {
                "id": memory.id,
                "caption": memory.caption,
                "content": memory.content,
                "emotional_tags": memory.emotional_tags,
                "suggested_tags": memory.suggested_tags,
                "sentiment_scores": memory.sentiment_scores,
                "timestamp": memory.timestamp,
                "file_location": file_location
            }
            
        except Exception as e:
            logging.error(f"Error uploading memory: {str(e)}")
            raise Exception(f"Failed to upload memory: {str(e)}")

    async def retrieve_memories(self, 
                              query: str = None, 
                              emotion: str = None, 
                              similar_to_id: int = None,
                              limit: int = 10) -> List[Dict]:
        try:
            if similar_to_id:
                # Get similar memories using vector similarity
                memory = self.db.query(Memory).filter(Memory.id == similar_to_id).first()
                if memory:
                    results = self.collection.query(
                        query_embeddings=[memory.text_embedding],
                        n_results=limit
                    )
                    memory_ids = [int(id) for id in results['ids'][0]]
                    memories = self.db.query(Memory).filter(Memory.id.in_(memory_ids)).all()
            elif emotion:
                # Filter by emotion tag
                memories = self.db.query(Memory).filter(
                    Memory.emotional_tags.contains([emotion])
                ).limit(limit).all()
            elif query:
                # Text search in captions and content
                memories = self.db.query(Memory).filter(
                    (Memory.caption.contains(query)) | 
                    (Memory.content.contains(query))
                ).limit(limit).all()
            else:
                memories = self.db.query(Memory).limit(limit).all()

            return [
                {
                    "id": memory.id,
                    "caption": memory.caption,
                    "content": memory.content,
                    "emotional_tags": memory.emotional_tags,
                    "timestamp": memory.timestamp,
                    "image_path": memory.image_path,
                    "sentiment_scores": memory.sentiment_scores
                }
                for memory in memories
            ]
        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            raise Exception(f"Failed to retrieve memories: {str(e)}")