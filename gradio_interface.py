import gradio as gr
import requests
from PIL import Image
import json

API_URL = "http://localhost:8000"

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

def search_memories(query):
    response = requests.get(f"{API_URL}/retrieve_memory/", 
                         params={"query": query})
    memories = response.json()
    
    # Prepare gallery items
    gallery_items = []
    if memories:
        for memory in memories:
            if "image_path" in memory:
                gallery_items.append((memory["image_path"], memory["caption"]))
    
    return gallery_items, memories

def chat_with_ai(user_input, history):
    response = requests.get(f"{API_URL}/chat/", 
                         params={"user_input": user_input})
    ai_response = response.json()
    
    # Update chat history
    history.append((user_input, ai_response["message"]))
    
    return history, ai_response.get("related_memories", [])

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Emotion Bank")
        
        with gr.Tabs():
            # Memory Upload Tab
            with gr.Tab("Upload Memory"):
                with gr.Column():
                    caption = gr.Textbox(label="Caption", placeholder="Enter a title for your memory")
                    content = gr.Textbox(label="Description", placeholder="Describe your memory...", lines=3)
                    emotional_tags = gr.Textbox(label="Emotional Tags", placeholder="happy, excited, grateful...")
                    image_file = gr.File(label="Upload Image", type="file")
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
    demo.launch()