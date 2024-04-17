import numpy as np
from som import SOM
import generate_photo
from collections import Counter


from graph import create_accuracy_graph, create_graphs, create_overlapping_graphs

if __name__ == '__main__':
    w,h = (64,64)
    min_fig_size=10
    num_shapes=5
    num_samples = 100
    class_names = ['Box', 'Circle', 'Triangle']
    
    
    generate_photo.genererate_pictures("data_set", num_samples, w, h, min_fig_size, num_shapes)
    
    # Обновите список folders, чтобы он содержал пути к изображениям для всех трех классов
    folders = ['./data_set/box', './data_set/circle', './data_set/triangle']
    
    # Генерация данных из изображений
    data = generate_photo.images_to_data_vector(folders)
    # Создание массива целевых значений для трех классов
    targets = np.array(num_samples * [0] + num_samples * [1] + num_samples * [2])  # Для каждого класса 30 изображений
    
    generate_photo.genererate_pictures("test_data_set", num_samples, w, h, min_fig_size, num_shapes)
    test_data_box = generate_photo.images_to_data_vector(['./test_data_set/box'])
    test_data_circle = generate_photo.images_to_data_vector(['./test_data_set/circle'])
    test_data_triangle = generate_photo.images_to_data_vector(['./test_data_set/triangle'])
    test_data = [test_data_box,test_data_circle,test_data_triangle]
    
    train_data = data
    train_targets = targets


    step_boost = 10
    max_epochs = 50000
    max_accuracy = 0.95
    combine_epochs_array=[]
    combine_accuracy_array=[]
    decay_array = ['hill','linear']
    for decay in decay_array:
        epochs = 10
        epohc_step = 0
        accuracy = 0.0
        epochs_array = []
        accuracy_array = []
        while epochs < max_epochs:
            epochs += epohc_step
            epohc_step += step_boost 

            # Инициализация SOM
            som = SOM(1, 3)  # Инициализация 1x3 SOM
            
            # Обучение SOM
            som.fit(train_data, epochs, save_e=False, interval=100, verbose=True, decay=decay)  
            
            #create_graphs(som, train_data, train_targets, class_names)

            # соседи точки данных
            #datapoint = generate_photo.image_to_vector(generate_photo.draw_boxes_random_rotated(w=w, h=h, fc=generate_photo.random_color(), mfs=min_fig_size, num_shapes=num_shapes))
            #print("Labels of neighboring datapoints: ", som.get_neighbors(datapoint, train_data, train_targets, d=0))
            figs_accuracy = []
            class_num_array = []
            for i in range(len(test_data)):
                
                
                predicted_targets = [som.winner(vector) for vector in test_data[i]]
                predicted_targets = [item[1] for item in predicted_targets]

                filtered_arr = [num for num in predicted_targets if num not in class_num_array]
                if filtered_arr:
                    counter = Counter(filtered_arr)
                    # Находим наиболее часто встречающуюся цифру и её количество
                    most_common_digit, count = counter.most_common(1)[0]
                    class_num_array.append(most_common_digit)
                    
                    figs_accuracy.append(count / len(test_data[i]))
                else:
                    figs_accuracy.append(0)
            

            accuracy = sum(figs_accuracy)/ len(figs_accuracy)    
                
            accuracy_array.append(accuracy)
            epochs_array.append(epochs)
            
            print(f'Accuracy: {accuracy}')
            print(f'epochs: {epochs}')
            
        combine_accuracy_array.append(accuracy_array)
        combine_epochs_array.append(epochs_array)
        
        create_accuracy_graph(epochs_array, accuracy_array, "epochs", decay,decay)
        create_graphs(som, train_data, train_targets, class_names, decay)
        
    create_overlapping_graphs(combine_epochs_array[0], combine_accuracy_array[0], decay_array[0], combine_epochs_array[0], combine_accuracy_array[1], decay_array[1], path="", for_name="epochs")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
    # Преобразование данных в пространство SOM
    generate_photo.genererate_pictures("data_set", num_samples, w, h, min_fig_size, num_shapes)

    
    # Генерация данных из изображений
    data = generate_photo.images_to_data_vector(folders)
    newdata = data
    transformed = som.transform(newdata)
    print("Old shape of the data:", newdata.shape)
    print("New shape of the data:", transformed.shape)
'''