from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models
import requests
from io import BytesIO


import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    './brain_tumor_dataset/train',
    target_size=(224, 224),  # MobileNet input size
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    './brain_tumor_dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential()

# Add MobileNet base
model.add(base_model)

# Add custom layers
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))  # num_classes is the number of your custom categories

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=50, validation_data=test_generator)

# Example for making predictions
img = tf.keras.preprocessing.image.load_img('./brain_tumor_dataset/test/yes/Y185.JPG', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
print(predictions)


app = Flask(__name__)
CORS(app)
api = Api(app)

@app.route('/predict', methods=['POST'])
def predict():

    req = request.get_json()
    print(req.get("url"))
    '''
    # Get the image file from the request
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Return the predicted class and probability
    '''
    response = {
        'class_index': "hello",
        'probability': "bye"
    }

    return jsonify(response)
    

@app.route('/predict_url', methods=['POST'])
def predict_url():
    # Get the image URL from the request
    data = request.get_json()
    image_url = data.get('url')
    print(image_url)

    # Fetch the image from the URL
    response = requests.get(image_url)
    print(response)
    img = tf.keras.preprocessing.image.load_img(BytesIO(response.content), target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet.preprocess_input(np.expand_dims(img_array, axis=0))

    # Make predictions
    predictions = model.predict(img_array)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(predictions)
    category = "unsure"
    if predicted_class_index == 0:
        category = "no tumor"
    elif predicted_class_index == 1:
        category = "tumor"

    # Return the predicted class and probability
    response = {
        'class_index': category,
        'probability': float(predictions[0][predicted_class_index])
    }

    return jsonify(response)


if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000)



    