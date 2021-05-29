## Цель проекта
Цель проекта состоит в том, чтобы из папки фото извлечь фотографии кошек, снятыx крупным планом, и собрать их в новую папку. 
## Как реализовано
С помощью сети Unet и претренированного энкодера делается семантическая сегментация каждой фотографии
(на выходе всего два класса - "принадлежит кошке" и "не принадлежит кошке". После сегментации подсчитывается, сколько процентов от фотографии составляют пиксели, относящиеся к кошке. Если кошка занимает больше 30 % фотографии, то она перемещается в новую папку. 
Тренировочный датасет, который использовался для данной задачи:
https://cocodataset.org/#home
(1480 картинок кошек и масок для них. Как загрузить датасет будет указано ниже)
В качестве loss функции мы исплользовали линейную комбинацию logloss и diceloss.

## Загрузка
Чтобы запустить приложение нужно запустить скрипт:
```bash  
chmod u+x starting.sh
bash starting.sh
```
Или выполнить по очереди следующие команды:
```bash  
git clone https://github.com/gatupovmikhail/ML_project  
cd ML_project  
conda env create --file gatupov.yml python=3.7.10 
conda activate gatupov  
chmod u+x down_test.sh  
bash down_test.sh  
pytnon3 app.py  
```
Скрипт **down_test.sh** Загрузит тестовый датасет (103 картинки, 10-15 минут) и веса обученной нейросети. 
После запуска программы **app.py** появится графическое окно. Через появившееся окно нужно войти **внутрь** дирректории **example_dataset** (Название папки должно появиться в поле окна "Select"). То, что фотографии не отображаются внутри окна, нормально. 
Также вы можете выбрать свою папку с фотографиями вместо предложенной.
В результате работы программы внутри выбранной папки появятся папки **photos_with_cats** и photos_without_cats. В первой папке будут фото кошек крупным планом.
Команды, описанные выше, нужно выполнить один раз. После этого приложение запускается командой:
```bash
pytnon3 app.py  
```


## Нейросеть
Сама нейросеть, даталоадер и "тренер" нейросети находятся в скрипте **my_net.ipynb**. Однако, чтобы заново потренировать модель нужно загрузить полный датасет данных с сайта COCO:
```bash
chmod u+x down_full.sh  
bash down_full.sh    
```
И загружаться он будет довольно долго. 
