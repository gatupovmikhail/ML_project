#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import torch
import torch.utils.data as D
import torch.nn as nn
import pandas as pd
import os
import albumentations as A
from torchvision import transforms as T
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet, FPN
import time
import shutil
import magic
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
print()
print('##### LOGS of app.py #####')
print()
print('Libraries imported successfully')

# Проверка доступа видеокарты
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = 'cpu'
print('Cpu или cuda?: ' + str(DEVICE))

def make_grid(shape, window=256, min_overlap=32):
    """
        Функкция, возвращающая массив размерами (N,4), число N - число ячеек,
        4 - координаты каждой ячекки x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]    
    return slices.reshape(nx*ny,4)

# Необходимые для работы переменные и функции
WINDOW=128
MIN_OVERLAP=16
NEW_SIZE=128
BATCH_SIZE = 5
DIR_OF_MODEL = 'CatPytorchModel'
name_model = 'model6'

trfm = A.Compose([
    A.Resize(NEW_SIZE,NEW_SIZE),
])

as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

def get_model(dir_of_model, name_model):
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES_NUMBER = 1
    ACTIVATION = 'sigmoid'

    # Предобученный Unet
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES_NUMBER,
        activation=ACTIVATION,
    )
    model.load_state_dict(torch.load(dir_of_model + '/' + name_model))
    return model


# Загрузка модели и отправка на GPU
model = get_model(DIR_OF_MODEL,name_model)
model.to(DEVICE)
model.eval()


@torch.no_grad()
def finding_of_cat(name_img, data_dir, window=WINDOW, min_overlap=MIN_OVERLAP,
                  batch_size=BATCH_SIZE, new_size=NEW_SIZE, make_grid=make_grid,trfm=trfm,
                  as_tensor=as_tensor,device=DEVICE):
    '''Каждую картинку прогоняет через нейросеть и определяет, какой процент
    пикселей принадлежит кошке. Возвращает этот процент'''
    image = io.imread(data_dir + '/' + name_img)
    slices = make_grid([image.shape[0],image.shape[1]], window=window,
                       min_overlap=min_overlap)
    image_tensor_slices = []
    for slc in slices:
        x1,x2,y1,y2 = slc
        img_slice = image[x1:x2,y1:y2]
        image_tensor = as_tensor(trfm(image=img_slice)['image'])
        image_tensor_slices.append(image_tensor)
    number_of_slices = len(image_tensor_slices)

    test_loader = D.DataLoader(
        image_tensor_slices, batch_size=batch_size, shuffle=False, num_workers=0)

    cat_pixels = 0
    for image_input in test_loader:
        image_input = image_input.to(device)
        output = model(image_input).cpu().detach().numpy()
        output[output >= 0.9] = 1
        output[output != 1] = 0
        for slc in output:
            cat_pixels += slc[0].sum()

    cat_pixels_persentage = float(cat_pixels/(new_size*new_size*number_of_slices))
    return cat_pixels_persentage


## Выбор дирректории через окно filebrowser

foto_dir = './' # здесь будет хранится выбранная дирректория
def c_open_dir_old():
    '''Функция для выбора дирректории через filebrowser'''
    dirr = os.getcwd()
    global foto_dir
    foto_dir = filedialog.askdirectory(parent=root, initialdir=dirr)
    root.destroy()

# Описание начального окна
root = tk.Tk()
style = ttk.Style(root)
style.theme_use("clam")
root.configure(bg=style.lookup('TFrame', 'background'))
text_label = 'You should choose directory wih your photos.' + \
'It"s normal, that you can see only directories (without files) in your directory.'+ \
'You should go INSIDE your directory (see the entry "Selection")'
text_window = ttk.Label(root, text=text_label)
ttk.Button(root, text="Open folder", command=c_open_dir_old).grid(row=2, column=0, \
                                                                  padx=4, pady=4, sticky='ew')
text_window.grid(row=1, column=0, padx=4, pady=4)
# Запуск окна
root.mainloop()

# Извлечение всех имен файлов из дирректории, фильтрация
# по форматам PNG и JPEG
print(f'Выбрана дирректория: \n {foto_dir}')
fotos = os.listdir(foto_dir)
number_files = len(fotos)
file_counter = 0
print('Определение форматов... ', end='')
fotos_filtered = []
for f in fotos:
    path_to_file = foto_dir + '/' + f
    if not(os.path.isdir(path_to_file)):
        file_type = magic.from_file(path_to_file)
        if file_type.split()[0] == 'PNG' or file_type.split()[0] == 'JPEG':
            fotos_filtered.append(f)
    file_counter += 1
print('Done')
# Переход в указанную дирректорию и создание папки с output:
old_directory = os.getcwd()
os.chdir(foto_dir)
dir_of_cats_foto = 'photos_with_cats'
dir_without_cats_foto = 'photos_without_cats'
if not os.path.exists(dir_of_cats_foto):
    os.mkdir(dir_of_cats_foto)
if not os.path.exists(dir_without_cats_foto):
    os.mkdir(dir_without_cats_foto)

#  Подготовка принтов для отслеживания процесса  
number_fotos = len(fotos_filtered)
foto_counter = 0
t_work = number_fotos*5
if t_work/60 < 1:
    print(f'Оценочное время работы программы: {t_work} с')
    print('(Может сильно варьироваться в зависимости от размеров фото)')
else:
    print(f'Оценочное время работы программы: {t_work} c ({int(t_work/60)} min)')
    print('(Может сильно варьироваться в зависимости от размеров фото)')
start_time = time.time()

# Проход каждого фото через сеть
theshold_percentage = 0.20  # какую часть фото должна занимать кошка
for f in fotos_filtered:
    percent = round(foto_counter/number_fotos,2)
    print('Обработано фото: {}/{}, {} %'.format(foto_counter, number_fotos, \
                                           int(percent*100)), end='\r')
    try:
        pers_cat = finding_of_cat(f, data_dir=foto_dir)
    except (ValueError, RuntimeError):
        foto_counter += 1
        print('Что-то не так с фото ', end='')
        print(f + ' Не могу обработать его.')
        continue
    if  pers_cat >= theshold_percentage:
        shutil.copy(f, dir_of_cats_foto)
    else:
        shutil.copy(f, dir_without_cats_foto)
    foto_counter += 1

# Информация о прошедшем процессе
print('Обработано фото: {}/{}, {} %'.format(foto_counter, number_fotos, 100)) 
print('Done!')
print(f'Реальное время работы: {round(time.time() - start_time,2)} c')
print(f'Фото кошек сохранены в папке {foto_dir + "/" + dir_of_cats_foto}')
os.chdir(old_directory)
