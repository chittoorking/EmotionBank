import gradio as gr
import requests
from PIL import Image
import json
import time

API_URL = "http://127.0.0.1:8000"

def check_api_connection():
    try:
        requests.get(f"{API_URL}/")
        return True
    except requests.exceptions.ConnectionError:
        return False

def handle_api_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to the backend server. Please ensure the FastAPI server is running."}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}
    return wrapper

@handle_api_error
def upload_memory(caption, content, emotional_tags, file):
    if file is None:
        return {"error": "Please upload an image"}
    
    tags = [tag.strip() for tag in emotional_tags.split(',')]
    files = {'file': file}
    data = {
        'caption': caption,
        'content': content,
        'emotional_tags': json.dumps(tags)
    }
    
    response = requests.post(f"{API_URL}/upload_memory/", 
                          files=files, 
                          data=data)
    return response.json()

@handle_api_error
def search_memories(query):
    response = requests.get(f"{API_URL}/retrieve_memories/", 
                         params={"query": query})
    memories = response.json()
    
    gallery_items = []
    if memories:
        for memory in memories:
            if "image_path" in memory:
                gallery_items.append((memory["image_path"], memory["caption"]))
    
    return gallery_items, memories

@handle_api_error
def chat_with_ai(user_input, history):
    response = requests.get(f"{API_URL}/chat/", 
                         params={"user_input": user_input})
    ai_response = response.json()
    
    history.append((user_input, ai_response["message"]))
    
    return history, ai_response.get("related_memories", [])

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        if not check_api_connection():
            gr.Markdown("⚠️ Warning: Cannot connect to backend server. Please start the FastAPI server first.")
        
        gr.Markdown("# Emotion Bank")
        
        with gr.Tabs():
            # Memory Upload Tab
            with gr.Tab("Upload Memory"):
                with gr.Column():
                    caption = gr.Textbox(label="Caption", placeholder="Enter a title for your memory")
                    content = gr.Textbox(label="Description", placeholder="Describe your memory...", lines=3)
                    emotional_tags = gr.Textbox(label="Emotional Tags", placeholder="happy, excited, grateful...")
                    image_file = gr.File(label="Upload Image", type="binary")
                    upload_button = gr.Button("Save Memory", variant="primary")
                    upload_output = gr.JSON(label="Upload Result")

            # Memory Retrieval Tab
            with gr.Tab("Browse Memories"):
                with gr.Row():
                    search_text = gr.Textbox(label="Search Memories", placeholder="Search by text or emotion...")
                    search_button = gr.Button("Search")
                
                with gr.Row():
                    memories_gallery = gr.Gallery(label="Found Memories")
                    memory_details = gr.JSON(label="Memory Details")

            # AI Companion Tab
            with gr.Tab("Chat & Reflect"):
                with gr.Column():
                    chat_history = gr.Chatbot(label="Chat History")
                    user_message = gr.Textbox(label="Your Message", placeholder="Type your message...")
                    chat_button = gr.Button("Send")
                    with gr.Accordion("Related Memories", open=False):
                        related_memories = gr.JSON(label="Related Memories")

        # Connect event handlers
        upload_button.click(upload_memory, 
                          inputs=[caption, content, emotional_tags, image_file],
                          outputs=upload_output)
        
        search_button.click(search_memories,
                          inputs=search_text,
                          outputs=[memories_gallery, memory_details])
        
        chat_button.click(chat_with_ai,
                        inputs=[user_message, chat_history],
                        outputs=[chat_history, related_memories])

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)  # Added share=True for public access