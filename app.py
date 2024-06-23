from scripts.Run import train_model
import gradio as gr

model = train_model()
demo = gr.Interface(model.__call__, "text", "text")
demo.launch(share=True)