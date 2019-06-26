#!/bin/bash

echo "Installing Python requirements"
virtualenv --no-site-packages -p /usr/bin/python3 env
source env/bin/activate
pip install -r requirements.txt

echo "Downloading pretrained model"
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth -P pretrained/
