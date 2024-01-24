from numba import cuda
import numpy as np
import os
import cv2
import math





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




kernel_noise = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]], dtype=np.float32) / 9.0


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

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
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



def preprocessing_images(input_folder, output_folder):



    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    input_files = os.listdir(input_folder)

    for input_file in input_files:

        input_image_path = os.path.join(input_folder, input_file)
        rgb_image = cv2.imread(input_image_path)


        if rgb_image is not None:

            gray_image = rgb_to_gray(rgb_image)


            noise_reduced_image = custom_noise_reduction(gray_image)


            edge_image = sobel_edge_detection(noise_reduced_image)


            output_file = os.path.splitext(input_file)[0] + '_edges.png'
            output_image_path = os.path.join(output_folder, output_file)
            cv2.imwrite(output_image_path, edge_image)



    print("Conversion, noise reduction, and edge detection completed.")