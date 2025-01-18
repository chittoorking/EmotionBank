from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Memory
from memory_handler import MemoryHandler
from emotion_tagging import EmotionAnalyzer
from ai_companion import AICompanion
from datetime import datetime
from database import SessionLocal, init_db
import uvicorn
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_FOLDER = "/tmp/memory_uploads"  # Designated temp folder

# Ensure the temporary folder exists
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = FastAPI(title="EmotionBank API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()
db = SessionLocal()

# Initialize components
emotion_analyzer = EmotionAnalyzer()
memory_handler = MemoryHandler(db, emotion_analyzer)
ai_companion = AICompanion(memory_handler)

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "EmotionBank API is running"}

@app.post("/upload_memory/")
async def upload_memory(
    file_path:str = Form(...) ,
    caption: str = Form(...),
    content: str = Form(...),
    emotional_tags: str = Form(...)
):
    logger.info(f"Received upload request - Caption: {caption}")
    try:
        # Parse emotional tags
        tags = json.loads(emotional_tags)
        
        # Create memory object
        memory = Memory(
            file_path=file_path,
            caption=caption,
            content=content,
            emotional_tags=tags,
            timestamp=datetime.now()
        )
                
        try:
            # Pass the file path to the memory handler
            result = await memory_handler.upload_memory(memory)
            logger.info("Upload successful")
            return result
        except:
            logger.error("Upload failed")
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.get("/retrieve_memories/")
async def retrieve_memories(query: str = None):
    try:
        memories = await memory_handler.retrieve_memories(query=query)
        return memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/")
async def chat_with_ai(user_input: str):
    try:
        response = await ai_companion.chat(user_input)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
