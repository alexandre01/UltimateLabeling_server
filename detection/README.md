# Object Detection

```
    ├── README.md
    ├── darknet            : Implementation of yolo in C++ (For training the model)
    ├── yolov3             : Implementation of yolo in PyTorch 0.41 (For inferring the images)
    └── download_files.sh  : Script to download the trained weights, example images and  dataset to train the model
```

# Introduction

The object detection is done by [Yolov3](https://pjreddie.com/darknet/yolo/). 

To train the model from scratch, please follow the following steps or use trained model, skip step 2. 

The aerail_dataset (about 7GB) to train the model is available [here](https://drive.google.com/open?id=1rUcUKc8Vgs8wERgDnG1FfHHDl8Q7hu-I).


## 1. Download files
Run `download_files.sh`. It will automatically download the trained weights, example images and  dataset to train the model and put them into corresponding directories.

```
$ chmod +x download_files.sh
$ ./download_files.sh 
```
## 2. Train the model:

In this project, I used the modified version of [darknet](https://github.com/AlexeyAB/darknet) from AlexeyAB. The original implementation of yolov3 in darknet is [here](https://github.com/pjreddie/darknet)

1. Compile darknet
   ```
   $ cd darknet
   $ make
   ```
   For more details of configuring Makefile, please see the README.md of [how-to-compile-on-linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux) in AlexeyAB's repository.

2. Start training
    ```
    $ ./darknet detector train yolo_cfg/topview-6.data yolo_cfg/topview-6.cfg weights/darknet53.conv.74
    ```
## 3. Inference on the images

In this part I used [pytorch-0.4-yolov3 : Yet Another Implimentation of Pytroch 0.41 or over and YoloV3](https://github.com/andy-yun/pytorch-0.4-yolov3) from andy-yun. This implementation is more easy to setup. However, the implementation of not 100% identical to the original yolov3, especially in training step. Therefore, I write `batch_detect.py` to utilize this implementation to do the inference only.

### For single image, run `detect.py`

**Example:**
```
$ python detect.py  yolo_cfg/topview-6-predict.cfg  weights/topview-final-6.weights data/example.jpg yolo_cfg/topview-6.names  
```



### For multiple images, run `batch_detect.py`

**Usage:**
```
usage: batch_detect.py [-h] [--weights WEIGHTS] [--cfg CFG] [--names NAMES]
                       [--src_path SRC_PATH] [--output_path OUTPUT_PATH]
                       [--save_anno] [--text_only]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS, -w WEIGHTS
                        path of the trained weights
  --cfg CFG, -c CFG     path of yolo cfg file
  --names NAMES, -n NAMES
                        path of class names
  --src_path SRC_PATH, -sp SRC_PATH
                        folder path to source images
  --output_path OUTPUT_PATH, -op OUTPUT_PATH
                        folder path to output images
  --save_anno, -s       Save result in to text file in yolo format
  --text_only, -t       Save result without images

```

**Example**
```
$ python batch_detect.py -w weights/topview-final-6.weights \
        -c yolo_cfg/topview-6-predict.cfg \
        -n yolo_cfg/topview-6.names \
        -sp data/frames \
        -op data/frames_pred \
        -s 
```
The example command above will infer all the images in `data/frames` and save the all annotations and images with bounding boxes in `data/frames_pred`



   











