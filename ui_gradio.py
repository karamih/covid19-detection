import gradio as gr
from prediction import predict

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='filepath'),
                    outputs=['label', 'number'],
                    examples=["covid-positive.jpg"])

demo.launch(debug=True)