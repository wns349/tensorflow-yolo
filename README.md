# TensorFlow-Yolo
Python + TensorFlow implementation of YOLO (You Only Look Once).

### Requirements
Refer to `requirements.txt`
- Python 3
- Numpy
- OpenCV
- TensorFlow


### YOLO Networks

For the time being, I've only impelemented yolo network for [tiny-yolo-voc](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny-voc.cfg) and [yolov2](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg).

A pre-trained weights file can be downloaded [here](https://pjreddie.com/media/files/yolov2-tiny-voc.weights) and [here](https://pjreddie.com/media/files/yolov2.weights), depending on the network.

### Quick Start

1. Installation

    ```bash
    $ git clone https://github.com/wns349/tensorflow-yolo

    # Install requirements. Hopefully, `pip` takes care of it all
    $ pip install -r requirements.txt
    ```


2. Download pre-trained weights file

    Download and keep it under `/bin` directory. (or anywhere else you'd like.)

3. Run prediction

    ```bash
    $ python ./predict.py --img_dir ./img --out_dir ./out --weights ./bin/yolov2.weights --names ./resource/coco.names --anchors ./resource/yolov2-coco.anchors --network v2 --threshold 0.5 --iou_threshold 0.5
    ```

    After a successful inference, a resulting image with bounding boxes is saved as `./out`.


### TODO
- Train custom objects for detection

##### References
I found the following projects/websites to be very helpful. Many thanks!
- [Darknet](https://github.com/pjreddie/darknet/)
- [YAD2K](https://github.com/allanzelener/YAD2K)
- [Darkflow](https://github.com/thtrieu/darkflow/)
- [MLBLR#yolov2](https://mlblr.com/includes/mlai/index.html#yolov2)
- [Implementing YOLO v3 in Tensorflow(TF-Slim)](https://github.com/mystic123/tensorflow-yolo-v3)
- [Training Object Detection (YOLOv2) from scratch...](https://towardsdatascience.com/training-object-detection-yolov2-from-scratch-using-cyclic-learning-rates-b3364f7e4755)
