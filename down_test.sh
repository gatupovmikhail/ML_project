#!/bin/bash
echo 'Downloading of example dataset' && \
wget https://data.cyverse.org/dav-anon/iplant/home/mikhailgatupov/example_dataset.zip && \
unzip example_dataset.zip && \
rm example_dataset.zip && \
echo ' Done' && \
echo 'Downloading of model' && \
wget https://data.cyverse.org/dav-anon/iplant/home/mikhailgatupov/CatPytorchModel.zip  && \
unzip CatPytorchModel.zip && \
rm CatPytorchModel.zip && \
echo 'Done' && \
exit 0

