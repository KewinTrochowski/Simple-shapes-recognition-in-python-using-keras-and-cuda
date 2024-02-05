from PIL import Image, ImageDraw
import random
import os
import numpy as np
from numba import cuda
import math
import shutil

def create_shape(shape, image_size=(100, 100), shape_size=50, rotation_angle=None):
    image = Image.new('RGB', image_size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    draw = ImageDraw.Draw(image)

    shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if shape == 'circle':
        bbox = [(image_size[0] - shape_size) // 2, (image_size[1] - shape_size) // 2,
                (image_size[0] + shape_size) // 2, (image_size[1] + shape_size) // 2]
        shape_image = Image.new('RGBA', image_size)
        shape_draw = ImageDraw.Draw(shape_image)
        shape_draw.ellipse(bbox, outline=shape_color, fill=None)
        if rotation_angle is not None:
            shape_image = shape_image.rotate(rotation_angle, resample=Image.BILINEAR,
                                             center=(image_size[0] // 2, image_size[1] // 2), fillcolor=(0, 0, 0, 0))
        image.paste(shape_image, (0, 0), shape_image)

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

    elif shape == 'triangle':
        x0, y0 = (image_size[0] - shape_size) // 2, (image_size[1] + shape_size) // 2
        x1, y1 = (image_size[0] + shape_size) // 2, (image_size[1] + shape_size) // 2
        x2, y2 = image_size[0] // 2, (image_size[1] - shape_size) // 2
        shape_image = Image.new('RGBA', image_size)
        shape_draw = ImageDraw.Draw(shape_image)
        shape_draw.polygon([(x0, y0), (x1, y1), (x2, y2)], outline=shape_color, fill=None)
        if rotation_angle is not None:
            shape_image = shape_image.rotate(rotation_angle, resample=Image.BILINEAR,
                                             center=(image_size[0] // 2, image_size[1] // 2), fillcolor=(0, 0, 0, 0))
        image.paste(shape_image, (0, 0), shape_image)

    return image

# CUDA kernel for creating a circle
@cuda.jit
def create_circle_kernel(image, shape_color, shape_size, background_color):
    i, j = cuda.grid(2)
    if i < image.shape[0] and j < image.shape[1]:
        image[i, j, 0] = background_color[0]
        image[i, j, 1] = background_color[1]
        image[i, j, 2] = background_color[2]
        x, y = i - image.shape[0] // 2, j - image.shape[1] // 2
        distance = math.sqrt(x ** 2 + y ** 2)
        if distance <= shape_size / 2:
            image[i, j, 0] = shape_color[0]
            image[i, j, 1] = shape_color[1]
            image[i, j, 2] = shape_color[2]
# CUDA kernel for creating a square
@cuda.jit
def create_square_kernel(image, shape_color, shape_size, background_color):
    i, j = cuda.grid(2)
    if i < image.shape[0] and j < image.shape[1]:
        # Set the background color for each pixel
        image[i, j, 0] = background_color[0]
        image[i, j, 1] = background_color[1]
        image[i, j, 2] = background_color[2]

        # Calculate half of the square's size for positioning
        half_size = shape_size // 2

        # Check if the current pixel is within the bounds of the square
        if (image.shape[0] // 2 - half_size) <= i < (image.shape[0] // 2 + half_size) and \
                (image.shape[1] // 2 - half_size) <= j < (image.shape[1] // 2 + half_size):
            # Set the square's color
            image[i, j, 0] = shape_color[0]
            image[i, j, 1] = shape_color[1]
            image[i, j, 2] = shape_color[2]

# Wrapper function to create the shape using CUDA kernels
@cuda.jit
def create_triangle_kernel(image, shape_color, shape_size, rotation_angle, background_color):
    i, j = cuda.grid(2)

    if i < image.shape[0] and j < image.shape[1]:
        image[i, j, 0] = background_color[0]
        image[i, j, 1] = background_color[1]
        image[i, j, 2] = background_color[2]

        x0, y0 = image.shape[0] // 2, image.shape[1] // 2 + shape_size // 2
        x1, y1 = image.shape[0] // 2 + shape_size // 2, image.shape[1] // 2 - shape_size // 2
        x2, y2 = image.shape[0] // 2 - shape_size // 2, image.shape[1] // 2 - shape_size // 2

        # Rotate vertices
        theta = (rotation_angle * math.pi) / 180.0
        x0_rot = (x0 - image.shape[0] // 2) * math.cos(theta) - (y0 - image.shape[1] // 2) * math.sin(theta) + image.shape[0] // 2
        y0_rot = (x0 - image.shape[0] // 2) * math.sin(theta) + (y0 - image.shape[1] // 2) * math.cos(theta) + image.shape[1] // 2

        x1_rot = (x1 - image.shape[0] // 2) * math.cos(theta) - (y1 - image.shape[1] // 2) * math.sin(theta) + image.shape[0] // 2
        y1_rot = (x1 - image.shape[0] // 2) * math.sin(theta) + (y1 - image.shape[1] // 2) * math.cos(theta) + image.shape[1] // 2

        x2_rot = (x2 - image.shape[0] // 2) * math.cos(theta) - (y2 - image.shape[1] // 2) * math.sin(theta) + image.shape[0] // 2
        y2_rot = (x2 - image.shape[0] // 2) * math.sin(theta) + (y2 - image.shape[1] // 2) * math.cos(theta) + image.shape[1] // 2

        if (i - x0_rot) * (y1_rot - y0_rot) - (x1_rot - x0_rot) * (j - y0_rot) > 0 and \
           (i - x1_rot) * (y2_rot - y1_rot) - (x2_rot - x1_rot) * (j - y1_rot) > 0 and \
           (i - x2_rot) * (y0_rot - y2_rot) - (x0_rot - x2_rot) * (j - y2_rot) > 0:
            image[i, j, 0] = shape_color[0]
            image[i, j, 1] = shape_color[1]
            image[i, j, 2] = shape_color[2]

@cuda.jit
def test_create_triangle_kernel(image, shape_color, shape_size, rotation_angle, background_color):
    i, j = cuda.grid(2)

    if i < image.shape[0] and j < image.shape[1]:
        x0, y0 = image.shape[0] // 2, image.shape[1] // 2 + shape_size // 2
        x1, y1 = image.shape[0] // 2 + shape_size // 2, image.shape[1] // 2 - shape_size // 2
        x2, y2 = image.shape[0] // 2 - shape_size // 2, image.shape[1] // 2 - shape_size // 2

        # Rotate vertices
        theta = (rotation_angle * math.pi) / 180.0
        x0_rot = (x0 - image.shape[0] // 2) * math.cos(theta) - (y0 - image.shape[1] // 2) * math.sin(theta) + image.shape[0] // 2
        y0_rot = (x0 - image.shape[0] // 2) * math.sin(theta) + (y0 - image.shape[1] // 2) * math.cos(theta) + image.shape[1] // 2

        x1_rot = (x1 - image.shape[0] // 2) * math.cos(theta) - (y1 - image.shape[1] // 2) * math.sin(theta) + image.shape[0] // 2
        y1_rot = (x1 - image.shape[0] // 2) * math.sin(theta) + (y1 - image.shape[1] // 2) * math.cos(theta) + image.shape[1] // 2

        x2_rot = (x2 - image.shape[0] // 2) * math.cos(theta) - (y2 - image.shape[1] // 2) * math.sin(theta) + image.shape[0] // 2
        y2_rot = (x2 - image.shape[0] // 2) * math.sin(theta) + (y2 - image.shape[1] // 2) * math.cos(theta) + image.shape[1] // 2

        # Draw lines between the vertices
        draw_line(image, x0_rot, y0_rot, x1_rot, y1_rot, shape_color)
        draw_line(image, x1_rot, y1_rot, x2_rot, y2_rot, shape_color)
        draw_line(image, x2_rot, y2_rot, x0_rot, y0_rot, shape_color)

        # Fill the triangle
        if (i - x0_rot) * (y1_rot - y0_rot) - (x1_rot - x0_rot) * (j - y0_rot) > 0 and \
           (i - x1_rot) * (y2_rot - y1_rot) - (x2_rot - x1_rot) * (j - y1_rot) > 0 and \
           (i - x2_rot) * (y0_rot - y2_rot) - (x0_rot - x2_rot) * (j - y2_rot) > 0:
            image[i, j, 0] = shape_color[0]
            image[i, j, 1] = shape_color[1]
            image[i, j, 2] = shape_color[2]

def draw_line(image, x0, y0, x1, y1, shape_color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        # Draw a line with 5 pixels thickness
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ii = int(x0 + di)
                jj = int(y0 + dj)
                if 0 <= ii < image.shape[0] and 0 <= jj < image.shape[1]:
                    image[ii, jj, 0] = shape_color[0]
                    image[ii, jj, 1] = shape_color[1]
                    image[ii, jj, 2] = shape_color[2]

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
def create_shape_cuda(shape, image_size=(100, 100), shape_size=50, rotation_angle=None):
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    block_size = (16, 16)
    grid_size = ((image_size[0] + block_size[0] - 1) // block_size[0], (image_size[1] + block_size[1] - 1) // block_size[1])

    if shape == 'circle':
        background_color = np.random.randint(0, 256, size=3)
        create_circle_kernel[grid_size, block_size](image, shape_color, shape_size, background_color)
    elif shape == 'square':
        background_color = np.random.randint(0, 256, size=3)
        random_border_color = np.random.randint(0, 256, 3)
        create_square_kernel[grid_size, block_size](image, shape_color, shape_size, background_color)
    elif shape == 'triangle':
        background_color = np.random.randint(0, 256, size=3)
        create_triangle_kernel[grid_size, block_size](image, shape_color, shape_size, rotation_angle, background_color)

    return Image.fromarray(image)

def oild_generate_dataset(num_images=1000, output_dir='shape_dataset/'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shapes = ['circle', 'square', 'triangle']
    for i in range(num_images):
        shape = random.choice(shapes)
        image = create_shape(shape, (100, 100), random.randint(20, 70), random.randint(0, 360))
        image.save(os.path.join(output_dir, f'{shape}_{i}.png'))

def generate_dataset(num_images=1000, output_dir='shape_dataset/'):
    # Clear the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    shapes = ['circle', 'square', 'triangle']
    for i in range(num_images):
        shape = random.choice(shapes)
        image = create_shape_cuda(shape, (100, 100), random.randint(20, 70), random.randint(0, 360))
        image.save(os.path.join(output_dir, f'{shape}_{i}.png'))


# Generate the dataset
generate_dataset()
