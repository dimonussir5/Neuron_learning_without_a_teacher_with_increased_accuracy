import os
import time
from matplotlib import pyplot as plt

def create_graphs(som, train_data, train_targets, class_names, path):
    # Создание директории для сохранения изображений, если она не существует
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(f'images/{path}'):
        os.makedirs(f'images/{path}')
    # Визуализация истории ошибок обучения
    #som.plot_error_history(filename='images/{path}/som_error.png')



    # Визуализация обученного представления с метками классов
    som.plot_point_map(train_data, train_targets, class_names, filename=f'images/{path}/som.png')
    
    # Визуализация плотности класса
    for t in range(len(class_names)): 
        som.plot_class_density(train_data, train_targets, t=t, name=class_names[t], colormap='viridis', filename=f'images/{path}/class_{class_names[t]}.png')
    
    # Визуализация карты расстояний
    som.plot_distance_map(colormap='Blues', filename=f'images/{path}/distance_map.png')
    


def create_accuracy_graph(array, accuracy_array, array_name, path = "", for_name = None):
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(f'images/{path}'):
        os.makedirs(f'images/{path}')
        
    filename = f'images/{path}/{array_name}.png'
    time.sleep(1)
# Создаем новое окно для графика
    plt.figure()

    positions = list(range(len(array)))


    # Строим график, используя позиции в качестве абсцисс
    plt.plot(positions, accuracy_array, marker='o')

    # Устанавливаем метки на оси абсцисс с использованием значений из array
    plt.xticks(positions, array)
    if(for_name):
        plt.title(f'Зависимость accuracy от {array_name}, для затухания {for_name}')
    else:
        plt.title(f'Зависимость accuracy от {array_name}')
    plt.xlabel(array_name)
    plt.ylabel('Точность (accuracy)')
    plt.grid(True)
    plt.savefig(filename) # Сохраняем график в файл
    plt.close() # Закрываем окно графика
    
    
import os
import time
import matplotlib.pyplot as plt

def create_overlapping_graphs(array1, accuracy_array1, array_name1, array2, accuracy_array2, array_name2, path="", for_name=None):
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(f'images/{path}'):
        os.makedirs(f'images/{path}')

    filename = f'images/{path}/overlapping_graphs.png'
    time.sleep(1)

    # Создаем новое окно для графика
    plt.figure(figsize=(10, 5))

    # Первый график (синий)
    plt.plot(array1, accuracy_array1, marker='o', color='blue', label=array_name1)

    # Второй график (красный) накладывается на первый
    plt.plot(array2, accuracy_array2, marker='o', color='red', label=array_name2)

    # Установка меток на оси абсцисс и ординат
    plt.xlabel(for_name)
    plt.ylabel('Точность (accuracy)')

    # Включение сетки
    plt.grid(True)

    # Установка легенды
    plt.legend()

    # Установка общего заголовка для графика

    plt.title(f'Зависимость accuracy от {array_name1} и {array_name2}')

    # Сохраняем график в файл
    plt.savefig(filename)
    plt.close()  # Закрываем окно графика