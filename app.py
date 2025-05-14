from flask import Flask, request, jsonify, send_file, render_template
import tensorflow as tf
from tensorflow.keras.applications import VGG19 #type:ignore
from tensorflow.keras.models import Model #type:ignore
import numpy as np
import cv2
import os
import time

app = Flask(__name__)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Load pre-trained VGG19 model
model = VGG19(include_top=False, weights='imagenet')
model.trainable = False

# Define content and style layers
style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
content_layer = 'block5_conv2'

content_model = Model(inputs=model.input, outputs=model.get_layer(content_layer).output)
style_models = [Model(inputs=model.input, outputs=model.get_layer(layer).output) for layer in style_layers]

def preprocess_image(image_path):
    """Load and preprocess an image for VGG19."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0).astype('float32')
    img[:, :, :, 0] -= 103.939
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 123.68
    return img

def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def content_loss(content, generated):
    return tf.reduce_mean(tf.square(content_model(content) - content_model(generated)))

def style_loss(style, generated):
    J_style = 0
    lam = 1.0 / len(style_models)
    for style_model in style_models:
        S = gram_matrix(style_model(style))
        G = gram_matrix(style_model(generated))
        J_style += tf.reduce_mean(tf.square(S - G)) * lam
    return J_style

def train_nst(content_path, style_path, iterations=50, alpha=10.0, beta=20.0):
    """Run the NST optimization loop."""
    content = preprocess_image(content_path)
    style = preprocess_image(style_path)
    generated = tf.Variable(content, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=5.0)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_loss(content, generated)
            J_style = style_loss(style, generated)
            J_total = alpha * J_content + beta * J_style
        gradients = tape.gradient(J_total, generated)
        optimizer.apply_gradients([(gradients, generated)])
    output_img = generated.numpy().squeeze()
    output_img[:, :, 0] += 103.939
    output_img[:, :, 1] += 116.779
    output_img[:, :, 2] += 123.68
    output_img = np.clip(output_img, 0, 255).astype('uint8')
    return output_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/nst', methods=['POST'])
def neural_style_transfer():
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'Missing required files'}), 400
    
    content_img = request.files['content']
    style_img = request.files['style']
    timestamp = str(int(time.time()))
    content_path = os.path.join('uploads', f'content_{timestamp}.jpg')
    style_path = os.path.join('uploads', f'style_{timestamp}.jpg')
    output_path = os.path.join('static', f'output_{timestamp}.jpg')
    
    content_img.save(content_path)
    style_img.save(style_path)
    output_image = train_nst(content_path, style_path)
    cv2.imwrite(output_path, output_image)
    
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
