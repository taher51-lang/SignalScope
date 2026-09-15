import gradio as gr
from backend.main import app as fastapi_app

# Hugging Face's "Gradio" spaces are 100% free and don't require Docker.
# To trick it into hosting our pure FastAPI backend, we create a tiny dummy Gradio UI
# and mount our real FastAPI app onto it.

def dummy_fn():
    return "SignalScope API is running. Please send requests to the /predict endpoint."

demo = gr.Interface(
    fn=dummy_fn, 
    inputs=None, 
    outputs="text",
    title="SignalScope API Backend"
)

# This merges our FastAPI backend with the required Gradio frontend.
# Hugging Face will run this file automatically.
app = gr.mount_gradio_app(fastapi_app, demo, path="/docs_ui")
