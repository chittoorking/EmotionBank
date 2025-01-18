from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Memory

DATABASE_URL = "sqlite:///./memories.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_memories():
    db = SessionLocal()
    memories = db.query(Memory).all()
    for memory in memories:
        print(f"ID: {memory.id}, Caption: {memory.caption}, Content: {memory.content}, Emotional Tags: {memory.emotional_tags}")

check_memories()
