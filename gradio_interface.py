import gradio as gr
import requests
import json
import os
import shutil
import time

API_URL = "http://127.0.0.1:8000"
UPLOAD_FOLDER = "./uploads/images"  # Directory to save uploaded files

def check_api_connection():
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
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
    
    try:
        # Ensure the upload directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        current_directory = os.getcwd() 

        # Construct the full path to the upload folder
        upload_folder = os.path.join(current_directory, "uploads\\images") 

        # Generate unique filename while preserving original filename
        timestamp = str(int(time.time()))
        original_filename = os.path.basename(file.name)
        temp_filename = f"{timestamp}_{original_filename}"

        temp_file_path = f"{upload_folder}\\{temp_filename}"  # Using `/` explicitly
        
        # Copy the uploaded file to temp location using simple copy
        shutil.copy2(file.name, temp_file_path)
        
        tags = [tag.strip() for tag in emotional_tags.split(',')]
        data = {
            'image_path':temp_file_path,
            'caption': caption,
            'content': content,
            'emotional_tags': json.dumps(tags)
        }
        
        response = requests.post(f"{API_URL}/upload_memory/", 
                              data=data)
        
        return response.json()
        
    except shutil.SameFileError:
        return {"error": "Source and destination file are the same"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@handle_api_error
def search_memories(query):
    response = requests.get(f"{API_URL}/retrieve_memories/", 
                         params={"query": query})
    memories = response.json()
    
    gallery_items = []
    if memories and isinstance(memories, list):
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
                    image_file = gr.File(label="Upload Image", type="filepath")  # Changed to 'binary'
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
    demo.launch(show_error=True)
