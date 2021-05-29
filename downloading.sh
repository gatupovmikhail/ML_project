#!/bin/bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip && \
unzip annotations_trainval2014.zip -d annotations_trainval2014 && \
rm annotations_trainval2014.zip && \
python3 download_data.py && \
echo 'Downloading of example dataset' && \
wget https://data.cyverse.org/dav-anon/iplant/home/mikhailgatupov/example_dataset.zip && \
unzip example_dataset.zip -d example_dataset && \
rm example_dataset.zip && \
echo ' Done' && \ 
exit 0
