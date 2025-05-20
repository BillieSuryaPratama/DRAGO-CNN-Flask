from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
model = load_model("Model.keras")
class_names = ['Antraknosa', 'Bercak Coklat', 'Hawar', 'Sehat', 'Busuk Batang', 'Kanker Batang']
img_height, img_width = 180, 180

@app.route('/predict', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_path = os.path.join('./image', imagefile.filename)
    imagefile.save(image_path)

    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(float(confidence), 2)
    })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
