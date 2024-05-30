from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np

import matplotlib.pyplot as plt
from keras.preprocessing.image import  img_to_array, load_img
from keras.applications.vgg19 import  preprocess_input
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import  img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

train_datagen = ImageDataGenerator(zoom_range= 0.5, shear_range= 0.3, horizontal_flip= True, preprocessing_function=preprocess_input)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



train=train_datagen.flow_from_directory(directory="New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
                                        target_size=(256,256),
                                        batch_size=32)

val=val_datagen.flow_from_directory(directory="New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
                                        target_size=(256,256),
                                        batch_size=32)


# Check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Call your prediction function here
            prediction_result = prediction(filepath)
            return render_template('index.html', prediction=prediction_result)
    return render_template('index.html')
from keras.models import load_model

model= load_model("best_model.h5")
ref=dict(zip(list(train.class_indices.values()),train.class_indices.keys()))

def prediction(path):
    original_img = load_img(path)
    img = load_img(path, target_size= (256,256))
    i = img_to_array(img)
    im = preprocess_input(i)
    img=np.expand_dims(im,axis=0)
    pred=np.argmax(model.predict(img))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    # Display the predicted image
    plt.subplot(1, 2, 2)
    plt.imshow(img[0])
    plt.title(f"Prediction: {ref[pred]}")
    plt.axis('off')
    
    plt.show()
    return ref[pred]

if __name__ == '__main__':
    app.run(debug=True)
