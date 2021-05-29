#!/usr/bin/env python
# coding: utf-8
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import torch.utils.data as D
import os
from progress.bar import IncrementalBar
import time
import sys

print()
print('##### LOGS of downloading.py #####')

# Папки, в которых располагаются .json 
# файлы для загрузки фото. Загружаются .sh скриптом
dataDir='annotations_trainval2014'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
DATA_DIR = 'coco_dataset/'

coco=COCO(annFile)

# Загружаем нужные нам картинки из базы данных. Предполагается, что уже есть
# json файлы, в которых содержатся URL картинок. (json загрузились раннее .sh скриптом)

catIds = coco.getCatIds(catNms=['cat'])  # отбор картинок с кошками (id категории)
imgIds = coco.getImgIds(catIds=catIds )  # отбор картинок с кошками (id картинок)
print(f'number of samples: {len(imgIds)}') # кол-во картинок с кошками

try:
    os.mkdir(DATA_DIR)
except FileExistsError:
    pass

amount_samples = len(imgIds)  # число картинок в датасете

bar = IncrementalBar('Downloading', max = amount_samples) # прогрессбар

# загрузка самих картинок и масок
for num_of_id in range(amount_samples):
    bar.next()
    # Картинок загрузка
    idd = imgIds[num_of_id]
    img = coco.loadImgs(idd)[0]
    I = io.imread(img['coco_url'])

    # Загрузка масок
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    mask = coco.annToMask(anns[0])
    plt.imsave(DATA_DIR + f'img_{num_of_id}', I, format='jpg')     # Сохранение картинки
    mask = (mask + 1)%2  # инвертирование цветов в маске! кот обозначен белым цветом.
    plt.imsave(DATA_DIR + f'mask_{num_of_id}', mask, format='jpg', cmap='Greys') # Сохранение маски
bar.finish()
print()
print('Dataset Successfully downloaded!')
