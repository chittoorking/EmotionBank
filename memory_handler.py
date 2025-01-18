import logging
from sqlalchemy.orm import Session
from fastapi import UploadFile
import os
import aiofiles
import numpy as np
from datetime import datetime
from typing import List, Dict
from models import Memory
import chromadb

class MemoryHandler:
    def __init__(self, db: Session, emotion_analyzer):
        self.db = db 
        self.emotion_analyzer = emotion_analyzer
        
        # Initialize ChromaDB for vector search
        self.vector_db = chromadb.PersistentClient(path="./vector_db")
        print(self.vector_db.list_collections())
        # Check if the collection 'text_memories' exists
        collections = self.vector_db.list_collections()
        if "text_memories" in collections:
            # If the collection exists, get it
            self.text_collection = self.vector_db.get_collection(name="text_memories")
        else:
            # If the collection doesn't exist, create a new one
            self.text_collection = self.vector_db.create_collection(name="text_memories")

        # Similarly for image collection
        if "image_memories" in collections:
            self.image_collection = self.vector_db.get_collection(name="image_memories")
        else:
            self.image_collection = self.vector_db.create_collection(name="image_memories")
        
        # Ensure directories exist
        os.makedirs("uploads/images", exist_ok=True)

    async def save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file and return the file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/images/{filename}"
        
        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)
        
        return file_path

    async def upload_memory(self, memory: Memory, file: UploadFile) -> Dict:
        try:
            # Save uploaded file
            file_path = await self.save_uploaded_file(file)
            
            # Analyze text content
            text_analysis = self.emotion_analyzer.analyze_text(memory.content)
            
            # Analyze image
            image_analysis = self.emotion_analyzer.analyze_image(file_path)
            
            # Combine analyses
            combined_analysis = self.emotion_analyzer.combine_analysis(
                text_analysis, image_analysis
            )
            
            # Update memory object with analysis results
            memory.image_path = file_path
            memory.text_embedding = text_analysis["text_embedding"]
            memory.image_embedding = image_analysis["image_embedding"]
            memory.suggested_tags = combined_analysis["primary_emotions"]
            memory.sentiment_scores = combined_analysis["emotion_scores"]
            
            # Save to database
            self.db.add(memory)
            self.db.commit()
            self.db.refresh(memory)
            
            # Add to vector databases
            self.text_collection.add(
                ids=[str(memory.id)],
                embeddings=[memory.text_embedding],
                metadatas=[{
                    "caption": memory.caption,
                    "emotional_tags": memory.emotional_tags,
                    "timestamp": memory.timestamp.isoformat()
                }]
            )
            
            self.image_collection.add(
                ids=[str(memory.id)],
                embeddings=[memory.image_embedding],
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
                "file_path": file_path
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
                    # Search both text and image collections
                    text_results = self.text_collection.query(
                        query_embeddings=[memory.text_embedding],
                        n_results=limit
                    )
                    image_results = self.image_collection.query(
                        query_embeddings=[memory.image_embedding],
                        n_results=limit
                    )
                    
                    # Combine and deduplicate results
                    memory_ids = list(set([
                        int(id) for id in text_results['ids'][0] + image_results['ids'][0]
                    ]))
                    memories = self.db.query(Memory).filter(
                        Memory.id.in_(memory_ids)
                    ).limit(limit).all()
                    
            elif emotion:
                # Filter by emotion tag
                memories = self.db.query(Memory).filter(
                    Memory.emotional_tags.contains([emotion])
                ).limit(limit).all()
                
            elif query:
                # Analyze query text for embedding
                query_analysis = self.emotion_analyzer.analyze_text(query)
                query_embedding = query_analysis["text_embedding"]
                
                # Search using query embedding
                results = self.text_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit
                )
                
                memory_ids = [int(id) for id in results['ids'][0]]
                memories = self.db.query(Memory).filter(
                    Memory.id.in_(memory_ids)
                ).all()
                
            else:
                memories = self.db.query(Memory).limit(limit).all()

            return [
                {
                    "id": memory.id,
                    "caption": memory.caption,
                    "content": memory.content,
                    "emotional_tags": memory.emotional_tags,
                    "suggested_tags": memory.suggested_tags,
                    "sentiment_scores": memory.sentiment_scores,
                    "timestamp": memory.timestamp,
                    "image_path": memory.image_path
                }
                for memory in memories
            ]
            
        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            raise Exception(f"Failed to retrieve memories: {str(e)}")