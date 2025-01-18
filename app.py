from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from models import Memory
from memory_handler import MemoryHandler
from emotion_tagging import EmotionAnalyzer
from ai_companion import AICompanion
from datetime import datetime
from database import SessionLocal, init_db, Base
import uvicorn
import json

app = FastAPI(title="EmotionBank API")

# Initialize database
init_db()
db = SessionLocal()

# Initialize components in correct order
emotion_analyzer = EmotionAnalyzer()
memory_handler = MemoryHandler(db, emotion_analyzer)
ai_companion = AICompanion(memory_handler)

@app.post("/upload_memory/")
async def upload_memory(
    file: UploadFile = File(...),
    caption: str = Form(...),
    content: str = Form(...),
    emotional_tags: str = Form(...)
):
    try:
        # Parse emotional tags
        try:
            tags = json.loads(emotional_tags)
        except json.JSONDecodeError:
            tags = [tag.strip() for tag in emotional_tags.split(',')]

        # Create memory object
        memory = Memory(
            caption=caption,
            content=content,
            emotional_tags=tags,
            timestamp=datetime.now()
        )
        
        # Upload memory and handle file
        result = await memory_handler.upload_memory(memory, file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrieve_memories/")
async def retrieve_memories(
    query: str = None,
    emotion: str = None,
    similar_to_id: int = None,
    limit: int = 10
):
    try:
        memories = await memory_handler.retrieve_memories(
            query=query,
            emotion=emotion,
            similar_to_id=similar_to_id,
            limit=limit
        )
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
    uvicorn.run(app, host="0.0.0.0", port=8000)