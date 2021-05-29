#!/bin/bash
echo 'Downloading of example dataset' && \
wget https://data.cyverse.org/dav-anon/iplant/home/mikhailgatupov/example_dataset.zip && \
unzip example_dataset.zip -d example_dataset && \
rm example_dataset.zip && \
echo ' Done' && \
echo 'Downloading of model' && \
echo 'Done' && \
exit 0

