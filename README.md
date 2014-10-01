Pedestrian Tracker demo
=======================

This project demonstrates a simple pedestrian tracker using HOG features support vector machine and a particle filter. It is not very fast as it runs a HOG detection for every particle, every frame. 


Files
=====

- trainer.py: run this to train a SVM on your training dataset
- tracker.py: track pedestrians in your test data using the previously trained SVM
- ParticleFilter.py: class that does the actual tracking
- dataset.py/dalal.py/iccv07.py: scripts that read in different datasets

library dependencies
====================

- OpenCV2
- numpy
- matplotlib
- sklearn


datasets
========

- ICCV07 Pedestrian dataset: http://www.vision.ee.ethz.ch/datasets_extra/iccv07-data.tar.gz
- DALAL Pedestrian dataset 

Screenshots
===========

![Screenshot](/screenshots/ScreenShot1.png?raw=true "Pedestrian Tracker showing the particles'")
![Screenshot](/screenshots/ScreenShot2.png?raw=true "Pedestrian Tracker showing the particles'")
