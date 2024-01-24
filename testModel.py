from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
import os
from preprocessing_images import preprocessing_images

def prepare_image(file_path, target_size=(100, 100)):

    image = load_img(file_path, target_size=target_size)  # Wczytanie i przeskalowanie obrazu
    image = img_to_array(image)  # Konwersja do tablicy
    image = np.expand_dims(image, axis=0)  # Dodanie wymiaru batch
    image /= 255.0  # Normalizacja
    return image

def predict_shape(model, image_folder, target_size=(100, 100)):

    shape_labels = ['circle', 'square', 'triangle']
    predictions = {}

    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtruje tylko obrazy
            file_path = os.path.join(image_folder, file_name)
            image = prepare_image(file_path, target_size=target_size)
            prediction = model.predict(image)
            predicted_shape = shape_labels[np.argmax(prediction)]
            predictions[file_name] = predicted_shape

    return predictions


loaded_model = load_model('my_model.keras')

preprocessing_images('test_Img', 'test_Img_preprocessed/')

test_images_folder = 'test_Img_preprocessed'

results = predict_shape(loaded_model, test_images_folder)
for file_name, shape in results.items():
    print(f"{file_name}: {shape}")