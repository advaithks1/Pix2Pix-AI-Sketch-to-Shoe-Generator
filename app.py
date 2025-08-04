import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, render_template, send_file
from PIL import Image
import io
import os

app = Flask(__name__)
    
IMG_SIZE = 128
final_model_path = "models/generator_final.h5"
best_model_path = "models/generator_epoch_24.h5"

final_generator = load_model(final_model_path, compile=False)
best_generator = load_model(best_model_path, compile=False)

def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_image(img_tensor):
    img = (img_tensor + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def tensor_to_image(tensor):
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()
    img = Image.fromarray(tensor)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['sketch']
        input_img = preprocess_image(file)
        final_output = final_generator(input_img, training=False)[0].numpy()
        best_output = best_generator(input_img, training=False)[0].numpy()

        final_img = postprocess_image(final_output)
        best_img = postprocess_image(best_output)
        input_img_raw = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        input_arr = np.array(input_img_raw).astype(np.uint8)

        # Save images to memory
        final_buf = tensor_to_image(final_img)
        best_buf = tensor_to_image(best_img)
        sketch_buf = tensor_to_image(input_arr)

        return render_template(
            'index.html',
            sketch_img='data:image/png;base64,' + encode_img(sketch_buf),
            final_img='data:image/png;base64,' + encode_img(final_buf),
            best_img='data:image/png;base64,' + encode_img(best_buf)
        )
    return render_template('index.html', sketch_img=None, final_img=None, best_img=None)

def encode_img(img_buf):
    import base64
    return base64.b64encode(img_buf.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)