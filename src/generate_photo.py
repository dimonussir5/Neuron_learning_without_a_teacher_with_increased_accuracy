import math
import os
import random
from shutil import rmtree
from PIL import Image, ImageDraw
import numpy as np


# Функция для генерации случайного цвета
def random_color():
    #return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return (255,255,255)


    # Функция для создания изображения и сохранения его в указанную директорию
def draw_triangles_random_rotated(w, h, fc, mfs, num_shapes):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for _ in range(num_shapes):
        side_length = random.randint(mfs, min(w, h) // 2 - 1)
        angle = random.uniform(0, 360)
        angle_rad = math.radians(angle)

        x1 = w // 2
        y1 = h // 2 - side_length // 2
        x2 = w // 2 - side_length // 2
        y2 = h // 2 + side_length // 2
        x3 = w // 2 + side_length // 2
        y3 = h // 2 + side_length // 2

        x1_rotated = int((x1 - w // 2) * math.cos(angle_rad) - (y1 - h // 2) * math.sin(angle_rad) + w // 2)
        y1_rotated = int((x1 - w // 2) * math.sin(angle_rad) + (y1 - h // 2) * math.cos(angle_rad) + h // 2)
        x2_rotated = int((x2 - w // 2) * math.cos(angle_rad) - (y2 - h // 2) * math.sin(angle_rad) + w // 2)
        y2_rotated = int((x2 - w // 2) * math.sin(angle_rad) + (y2 - h // 2) * math.cos(angle_rad) + h // 2)
        x3_rotated = int((x3 - w // 2) * math.cos(angle_rad) - (y3 - h // 2) * math.sin(angle_rad) + w // 2)
        y3_rotated = int((x3 - w // 2) * math.sin(angle_rad) + (y3 - h // 2) * math.cos(angle_rad) + h // 2)

        draw.polygon([(x1_rotated, y1_rotated), (x2_rotated, y2_rotated), (x3_rotated, y3_rotated)], fill=fc, outline=(0, 0, 0))
    return img

def draw_boxes_random_rotated(w, h, fc, mfs, num_shapes):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for _ in range(num_shapes):
        side_length = random.randint(mfs, min(w, h) // 2 - 1)
        angle = random.uniform(0, 360)
        angle_rad = math.radians(angle)

        x_center = random.randint(side_length // 2, w - side_length // 2)
        y_center = random.randint(side_length // 2, h - side_length // 2)

        half_length = side_length // 2
        half_width = side_length // 2

        x1_rotated = int(x_center - half_length * math.cos(angle_rad) + half_width * math.sin(angle_rad))
        y1_rotated = int(y_center - half_length * math.sin(angle_rad) - half_width * math.cos(angle_rad))
        x2_rotated = int(x_center + half_length * math.cos(angle_rad) + half_width * math.sin(angle_rad))
        y2_rotated = int(y_center + half_length * math.sin(angle_rad) - half_width * math.cos(angle_rad))
        x3_rotated = int(x_center + half_length * math.cos(angle_rad) - half_width * math.sin(angle_rad))
        y3_rotated = int(y_center + half_length * math.sin(angle_rad) + half_width * math.cos(angle_rad))
        x4_rotated = int(x_center - half_length * math.cos(angle_rad) - half_width * math.sin(angle_rad))
        y4_rotated = int(y_center - half_length * math.sin(angle_rad) + half_width * math.cos(angle_rad))

        draw.polygon([(x1_rotated, y1_rotated), (x2_rotated, y2_rotated), (x3_rotated, y3_rotated), (x4_rotated, y4_rotated)], fill=fc, outline=(0, 0, 0))
    return img

def draw_circles_random_rotated(w, h, fc, mfs, num_shapes):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for _ in range(num_shapes):
        r = random.randint(mfs // 2, min(w, h) // 2 - 1)
        angle = random.uniform(0, 360)
        angle_rad = math.radians(angle)

        x = random.randint(r, w - r)
        y = random.randint(r, h - r)

        x_rotated = int((x - w // 2) * math.cos(angle_rad) - (y - h // 2) * math.sin(angle_rad) + w // 2)
        y_rotated = int((x - w // 2) * math.sin(angle_rad) + (y - h // 2) * math.cos(angle_rad) + h // 2)

        draw.ellipse((x_rotated - r, y_rotated - r, x_rotated + r, y_rotated + r), fill=fc, outline=(0, 0, 0))
    return img

def genererate_pictures(_path, num_samples,w,h,min_fig_size,num_shapes):


    

    if os.path.exists(_path): 
        rmtree(_path)
    os.makedirs(_path)
    _folder = os.path.join(_path, "box")
    if not os.path.exists(_folder): os.makedirs(_folder)
    _folder = os.path.join(_path, "circle")
    if not os.path.exists(_folder): os.makedirs(_folder)
    _folder = os.path.join(_path, "triangle")
    if not os.path.exists(_folder): os.makedirs(_folder)



    for i in range(num_samples):
        # Генерация прямоугольника с одним из 7 цветов
        img_box = draw_boxes_random_rotated(w=w, h=h, fc=random_color(), mfs=min_fig_size, num_shapes=num_shapes)
        img_box.save(os.path.join(os.path.join(_path, "box"), f"box-{i}.png"))

        # Генерация круга с одним из 7 цветов
        img_circle = draw_circles_random_rotated(w=w, h=h, fc=random_color(), mfs=min_fig_size, num_shapes=num_shapes)
        img_circle.save(os.path.join(os.path.join(_path, "circle"), f"circle-{i}.png"))

        # Генерация круга с одним из 7 цветов
        img_treangle = draw_triangles_random_rotated(w=w, h=h, fc=random_color(), mfs=min_fig_size,num_shapes=num_shapes)
        img_treangle.save(os.path.join(os.path.join(_path, "triangle"), f"triangle-{i}.png"))
        


def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path).convert('L')  # Открываем изображение и преобразуем в оттенки серого
    resized_image = image.resize((new_width, new_height))  # Изменяем размер изображения
    return resized_image


def binary_vector_to_image(binary_vector, width, height):
    # Преобразование значений вектора обратно в значения пикселей
    binary_vector = np.where(binary_vector == 1, 255, 0)
    
    # Изменение размерности массива до двумерного массива (изображения)
    image_array = binary_vector.reshape((height, width)).astype(np.uint8)
    
    # Создание изображения из массива пикселей
    image = Image.fromarray(image_array)
    return image


def add_noise(image, noise_level):

        img_array = np.array(image.convert('L'))
        # Генерация случайных координат для добавления шума
        salt_and_pepper = np.random.rand(*img_array.shape)
        
        # Добавление salt-and-pepper шума к изображению
        img_array[salt_and_pepper < noise_level/2] = 0
        img_array[salt_and_pepper > 1 - noise_level/2] = 255
        
        # Преобразование массива обратно в изображение PIL
        noisy_image = Image.fromarray(img_array)
        return noisy_image


def get_noisy_picture(from_, w, h, noise_level):
    image = Image.open(from_)
    image = image.resize((w, h))  # Изменяем размер изображения
    noisy_image = add_noise(image, noise_level)
    return noisy_image



# Функция для преобразования изображения в вектор признаков
def image_to_vector(image):
    # Преобразование изображения в оттенки серого и уменьшение его размера
    image = image.convert('L') 
    # Преобразование изображения в вектор
    vector = np.array(image).flatten()
    # Нормализация вектора
    vector = vector / 255.0
    return vector


def images_to_data_vector(folders):
    # Загрузка и преобразование изображений в векторы признаков
    vectors = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Поддержка различных форматов изображений
                image_path = os.path.join(folder, filename)
                image = Image.open(image_path)
                vector = image_to_vector(image)
                vectors.append(vector)

    # Объединение всех векторов в одну большую матрицу
    data = np.vstack(vectors)
    return data
