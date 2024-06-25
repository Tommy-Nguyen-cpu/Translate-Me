from scripts.Run import load_model
import gradio as gr

model = load_model("trained_model/")
def get_output(sentence, max_word_outputs):
    return model(sentence, max_word_outputs).numpy().decode("utf-8")

demo = gr.Interface(get_output, ["text", "number"], "text")
demo.launch(share=True)