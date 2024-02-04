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

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop

import tensorflow as tf

pneumonia_train = ImageDataGenerator(rescale = 1/255)
pneumonia_validation = ImageDataGenerator(rescale = 1/255)

pneumonia_train_dataset = pneumonia_train.flow_from_directory("./pneumonia data/train", 
                                          target_size = (400,400), 
                                          batch_size = 16, 
                                          class_mode = 'binary')

pneumonia_validation_dataset = pneumonia_validation.flow_from_directory("./pneumonia data/test", 
                                          target_size = (400,400), 
                                          batch_size = 16, 
                                          class_mode = 'binary')

pneumonia_model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape  = (400,400,3)),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation = 'relu'),
tf.keras.layers.Dense(1,activation = 'sigmoid')
]
)

pneumonia_model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

model_fit = pneumonia_model.fit(pneumonia_train_dataset, 
                      steps_per_epoch=16, 
                      epochs=5,
                      validation_data=pneumonia_validation_dataset)

arthritis_train_datagen = ImageDataGenerator(rescale = 1/255)
arthritis_validation_datagen = ImageDataGenerator(rescale = 1/255)

arthritis_train_dataset = arthritis_train_datagen.flow_from_directory("./arthritis_data/train", 
                                          target_size = (400,400), 
                                          batch_size = 32, 
                                          class_mode = 'binary')

arthritis_validation_dataset = arthritis_validation_datagen.flow_from_directory("./arthritis_data/validate", 
                                          target_size = (400,400), 
                                          batch_size = 32, 
                                          class_mode = 'binary')

arthritis_model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape  = (400,400,3)),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation = 'relu'),
tf.keras.layers.Dense(1,activation = 'sigmoid')
]
)

arthritis_model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

arthritis_model_fit = arthritis_model.fit(arthritis_train_dataset, 
                      steps_per_epoch=50, 
                      epochs=5,
                      validation_data=arthritis_validation_dataset)


brain_train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

brain_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

brain_train_generator = brain_train_datagen.flow_from_directory(
    './brain_tumor_dataset/train',
    target_size=(224, 224),  # MobileNet input size
    batch_size=32,
    class_mode='categorical'
)

brain_test_generator = brain_test_datagen.flow_from_directory(
    './brain_tumor_dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

brain_base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in brain_base_model.layers:
    layer.trainable = False

brain_model = models.Sequential()

# Add MobileNet base
brain_model.add(brain_base_model)

# Add custom layers
brain_model.add(layers.GlobalAveragePooling2D())
brain_model.add(layers.Dense(256, activation='relu'))
brain_model.add(layers.Dropout(0.5))
brain_model.add(layers.Dense(2, activation='softmax'))  # num_classes is the number of your custom categories

brain_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

brain_model.fit(brain_train_generator, epochs=5, validation_data=brain_test_generator)

# Example for making predictions
img = tf.keras.preprocessing.image.load_img('./brain_tumor_dataset/test/yes/Y185.JPG', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = brain_model.predict(img_array)
print(predictions)


app = Flask(__name__)
CORS(app)
api = Api(app)

    

@app.route('/predict_brain_tumor', methods=['POST'])
def predict_brain_tumor():
    # Get the image URL from the request
    data = request.get_json()
    image_url = data.get('url')
    print(image_url)

    # Fetch the image from the URL
    response = requests.get(image_url)
    img = tf.keras.preprocessing.image.load_img(BytesIO(response.content), target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet.preprocess_input(np.expand_dims(img_array, axis=0))

    # Make predictions
    predictions = brain_model.predict(img_array)

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

@app.route('/predict_arthritis', methods=['POST'])
def predict_arthritis():
    # Get the image URL from the request
    data = request.get_json()
    image_url = data.get('url')

    # Fetch the image from the URL
    response = requests.get(image_url)

    img = tf.keras.preprocessing.image.load_img(BytesIO(response.content), target_size=(400, 400))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Rescale the image

    # Make predictions
    predictions = arthritis_model.predict(img_array)

    # Interpret the predictions as probabilities
    probability = predictions[0][0]
    # Make predictions
    val = "unsure"
    percent = 0
    if probability > 0.5:
        val = "arthritis"
        percent = probability
    elif probability <= 0.5:
        val = "not arthritis"
        percent = 1 - probability
    # Return the predicted class and probability
    response = {
        'class_index': str(val),
        'probability': str(percent)
    }

    return jsonify(response)

@app.route('/predict_pneumonia', methods=['POST'])
def predict_pneumonia():
    # Get the image URL from the request
    data = request.get_json()
    image_url = data.get('url')

    # Fetch the image from the URL
    response = requests.get(image_url)

    img = tf.keras.preprocessing.image.load_img(BytesIO(response.content), target_size=(400, 400))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Rescale the image

    # Make predictions
    predictions = pneumonia_model.predict(img_array)

    # Interpret the predictions as probabilities
    probability = predictions[0][0]
    # Make predictions
    val = "unsure"
    percent = 0
    if probability > 0.5:
        val = "pneumonia"
        percent = probability
    elif probability <= 0.5:
        val = "not pneumonia"
        percent = 1 - probability
    # Return the predicted class and probability
    response = {
        'class_index': str(val),
        'probability': str(percent)
    }

    return jsonify(response)


if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000)



    