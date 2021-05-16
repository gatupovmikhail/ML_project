#!/bin/bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip && \
unzip annotations_trainval2014.zip -d annotations_trainval2014 && \
rm annotations_trainval2014.zip && \
exit 0
