from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential
from preprocessing_images import preprocessing_images
from labeling import create_csv_labels


preprocessing_images('shape_dataset/', 'output_images/')

create_csv_labels('output_images/', 'labels.csv')

csv_file = 'labels.csv'
image_dir = 'output_images/'

# Wczytanie etykiet
labels = pd.read_csv(csv_file)

# Podział danych na zbiór uczący i testowy
train_df, test_df = train_test_split(labels, test_size=0.2)

# Generator danych (z augmentacją dla zbioru uczącego)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Przygotowanie generatorów
train_generator = train_datagen.flow_from_dataframe(train_df, directory=image_dir, x_col='filename', y_col='label', target_size=(100, 100), class_mode='categorical', batch_size=32)
test_generator = test_datagen.flow_from_dataframe(test_df, directory=image_dir, x_col='filename', y_col='label', target_size=(100, 100), class_mode='categorical', batch_size=32)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax') 
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(train_generator, validation_data=test_generator, epochs=100, verbose=1)

# Ocenianie modelu
loss, accuracy = model.evaluate(test_generator)
model.save('my_model.keras')
print(f'Test accuracy: {accuracy*100:.2f}%')