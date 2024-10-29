import gradio as gr
import numpy as np
from PIL import Image
from keras.models import load_model


clasiffier = load_model('../model_fs.keras')
denoise = load_model('../autoencoder_gaussian_noise.keras')

def greet(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.resize((180, 180))
    pred = clasiffier.predict(np.array([img]))

    if pred[0] > 0.5:
        pred_denoise = denoise.predict(np.array([img])/255)
        pred_denoise = pred_denoise[0]
    else:
        pred_denoise = img


    return {'Clear':(1-pred[0]), 'Noisy':pred[0]}, pred_denoise

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Image(type="filepath")],
    outputs=[gr.Label(), gr.Image()],
)

demo.launch()
