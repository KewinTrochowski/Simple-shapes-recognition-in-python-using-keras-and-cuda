# Simple shapes (triangle, square, circle) recognition using Keras and Cuda in Python


Kewin Trochowski 188860

Michał Zarzycki 184856

## Introduction

Our project was being created in a few main steps. Firstly, we had to created our database. Next we creatde some preprocessing function using CUDA from numba in order to prepare images to train our model. Then we created simple neutral network using Keras library and finally we tested it on some random images. 


## Creating database 

In order to create database we just generate 1000 images of 3 shapes. Each image was generated with random size of image, random rotation and random colors. Code below shows part of a script for generating squares:

```

    elif shape == 'square':
        bbox = [(image_size[0] - shape_size) // 2, (image_size[1] - shape_size) // 2,
                (image_size[0] + shape_size) // 2, (image_size[1] + shape_size) // 2]
        shape_image = Image.new('RGBA', image_size)
        shape_draw = ImageDraw.Draw(shape_image)
        shape_draw.rectangle(bbox, outline=shape_color, fill=None)
        if rotation_angle is not None:
            shape_image = shape_image.rotate(rotation_angle, resample=Image.BILINEAR,
                                             center=(image_size[0] // 2, image_size[1] // 2), fillcolor=(0, 0, 0, 0))
        image.paste(shape_image, (0, 0), shape_image)
```

All images are saved to one folder. For generating images and saving them we used PIL.
Next we had to labeled the images and we've done that by creating a csv file:



```

  def create_csv_labels(output_dir, csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])
        for filename in os.listdir(output_dir):
            if filename.endswith('.png'):
                label = filename.split('_')[0] 
                writer.writerow([filename, label])

```

Examples of generated images:

![circle_30](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/f19c7c78-1a86-48aa-bbb9-bf5e94ce426f)
![square_703](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/48537951-229c-4e08-b009-43379762ace3)
![triangle_284](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/4916bf5e-45af-4213-9e9f-1f2a35a906e6)



## Preprocessing images

Now, when we have a database we can preprocessed those images. We decided to use 3 methods: noise reduction, grey scaling and edge detection. We use CUDA for those operations in order to speed them up.
Noise reduction:

```

@cuda.jit
def custom_noise_reduction_kernel(input_image, output_image):
    x, y = cuda.grid(2)

    if x < input_image.shape[0] and y < input_image.shape[1]:
        if x > 0 and x < input_image.shape[0] - 1 and y > 0 and y < input_image.shape[1] - 1:


            pixel_sum = 0.0


            for i in range(3):
                for j in range(3):
                    pixel_sum += input_image[x - 1 + i, y - 1 + j] * kernel_noise[i, j]


            output_image[x, y] = pixel_sum

def custom_noise_reduction(input_image):

    output_image = np.empty_like(input_image)


    threadsperblock = (16, 16)
    blockspergrid_x = (input_image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (input_image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    custom_noise_reduction_kernel[blockspergrid, threadsperblock](input_image, output_image)

    return output_image

```

Grayscaling:

```

@cuda.jit
def rgb_to_gray_kernel(input_image, output_image):
    x, y = cuda.grid(2)

    if x < input_image.shape[0] and y < input_image.shape[1]:

        r = input_image[x, y, 0]
        g = input_image[x, y, 1]
        b = input_image[x, y, 2]


        gray_value = 0.299 * r + 0.587 * g + 0.114 * b


        output_image[x, y] = gray_value


def rgb_to_gray(input_image):

    output_image = np.empty((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)


    threadsperblock = (16, 16)
    blockspergrid_x = (input_image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (input_image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    rgb_to_gray_kernel[blockspergrid, threadsperblock](input_image, output_image)

    return output_image

```

Edge detection:

```
@cuda.jit
def sobel_edge_detection_kernel(input_image, output_image, sobel_x, sobel_y):
    x, y = cuda.grid(2)

    if x < input_image.shape[0] and y < input_image.shape[1]:
        if x > 0 and x < input_image.shape[0] - 1 and y > 0 and y < input_image.shape[1] - 1:

            gradient_x = 0.0
            gradient_y = 0.0


            for i in range(3):
                for j in range(3):
                    gradient_x += input_image[x - 1 + i, y - 1 + j] * sobel_x[i, j]


            for i in range(3):
                for j in range(3):
                    gradient_y += input_image[x - 1 + i, y - 1 + j] * sobel_y[i, j]


            gradient_magnitude = math.sqrt(gradient_x ** 2 + gradient_y ** 2)


            output_image[x, y] = gradient_magnitude


def sobel_edge_detection(input_image):

    output_image = np.empty_like(input_image)


    threadsperblock = (16, 16)
    blockspergrid_x = (input_image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (input_image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    sobel_edge_detection_kernel[blockspergrid, threadsperblock](input_image, output_image, sobel_x, sobel_y)

    return output_image


```
We used sobel edge detection algorithm because of its simplicity in implementation.

Examples of preprocessed images:

![circle_467_edges](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/5e10a046-7f11-4e00-ae59-1ac4e3cdd046)
![square_514_edges](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/57e8f6cb-ee6d-453f-b753-80b7fb34995f)
![triangle_787_edges](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/6cec3f75-88e4-478f-a96f-4a78d964cb6e)



## Creating Neutral Network

First, we organize data into training set and testing set.

```

train_df, test_df = train_test_split(labels, test_size=0.2)

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df, directory=image_dir, x_col='filename', y_col='label', target_size=(100, 100), class_mode='categorical', batch_size=32)
test_generator = test_datagen.flow_from_dataframe(test_df, directory=image_dir, x_col='filename', y_col='label', target_size=(100, 100), class_mode='categorical', batch_size=32)


```
Model used in training:

```
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

```

## Training our model

For training we used Adam optimizer and categorical cross-entropy loss function.

```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=test_generator, epochs=100, verbose=1)

```
We chose 100 epochs for our training as it resulted in high test accuracy: 97.5%

![image](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/1323b9ed-0268-49b1-8db9-6a19d1e29bd3)



## Testing model

We prepared 6 images to test our trained model. The images looks like this:

![kolo](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/7e52ba02-4553-4e80-9806-ec5b135f5512)
![kolo2](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/7baf6d2e-e6f8-4fbe-b023-2eb90ff8ce43)
![kwadrat3](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/83706ddd-5c6a-4d5f-a086-2de69543e4d2)
<img src="https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/8319be29-17fe-41d3-9c14-f9a4a75348d3" width="100" height="100">
<img src="https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/d8e90e45-c77f-424b-bff8-027edcb89c06" width="100" height="100">
<img src="https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/08277648-3982-4638-8cbc-ecfe23e2f57f" width="100" height="100">

The result we get:

![image](https://github.com/KewinTrochowski/Simple-shapes-recognition-in-python-using-keras-and-cuda/assets/106476589/4a15d4f6-c642-4305-bf04-7953a7501bc4)


