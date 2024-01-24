from PIL import Image, ImageDraw
import random
import os


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

# Pozostała część kodu bez zmian


def generate_dataset(num_images=1000, output_dir='shape_dataset/'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shapes = ['circle', 'square', 'triangle']
    for i in range(num_images):
        shape = random.choice(shapes)
        image = create_shape(shape, (100, 100), random.randint(20, 70), random.randint(0, 360))
        image.save(os.path.join(output_dir, f'{shape}_{i}.png'))


# Generate the dataset
generate_dataset()
