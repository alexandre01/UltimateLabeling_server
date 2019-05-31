#!/bin/bash

echo "Installing Python requirements"
virtualenv -p /usr/bin/python3 env
source env/bin/activate
pip install -r requirements.txt

echo "Downloading pretrained model"
gfileid="1MyNYp-APTvHTwUke7YA7zIqAfjnx5bYA"
destination_dir="./"
file_name="object_detection.tar.gz"
destination_path="${destination_dir}${file_name}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${gfileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${gfileid}" -o ${destination_path}
tar -zxf $file_name


folder="object_detection"
mv ${folder}/data/ yolov3/data/
rm -rf yolov3/weights yolov3/yolo_cfg yolov3/data
rm -rf darknet/weights darknet/yolo_cfg
cp -R ${folder}/weights yolov3/
cp -R ${folder}/yolo_cfg yolov3/
cp -R ${folder}/weights darknet/
cp -R ${folder}/yolo_cfg darknet/
rm -rf $file_name $folder cookie
