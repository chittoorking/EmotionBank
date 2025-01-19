from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
import datetime

DATABASE_URL = "sqlite:///./memories.db"

# Database connection and session setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Memories table
class Memory(Base):
    __tablename__ = "memories"
    id = Column(Integer, primary_key=True, index=True)
    caption = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    emotional_tags = Column(Text, nullable=True)  # JSON stored as TEXT
    suggested_tags = Column(Text, nullable=True)  # JSON stored as TEXT
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image_path = Column(String, nullable=True)
    text_embedding = Column(Text, nullable=True)  # JSON stored as TEXT
    image_embedding = Column(Text, nullable=True)  # JSON stored as TEXT
    sentiment_scores = Column(Text, nullable=True)  # JSON stored as TEXT

# Define ChatHistory table
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_input = Column(Text, nullable=True)
    ai_response = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    related_memory_ids = Column(Text, nullable=True)  # JSON stored as TEXT

from sqlalchemy import inspect

def check_tables():
    try:
        # Create an inspector object to inspect the database
        inspector = inspect(engine)
        
        # Get the table names
        table_names = inspector.get_table_names()
        
        # Log the table names
        logging.info(f"Table names in the database: {table_names}")
        return table_names
    except Exception as e:
        logging.error(f"Error fetching table names: {str(e)}")

# Initialize database and create tables
def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        check_tables()
        logging.info("Database tables created successfully.")
        logging.info("Tables created: memories, chat_history")
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")

