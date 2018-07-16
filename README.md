# TensorFlow-Yolo
Python + TensorFlow implementation of YOLO (You Only Look Once).

### Requirements
Refer to `requirements.txt`
- Python 3
- Numpy
- OpenCV
- TensorFlow


### YOLO Networks

For the time being, I've only impelemented yolo network for [tiny-yolo-voc](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny-voc.cfg).

A pre-trained weights file can be downloaded [here](https://pjreddie.com/media/files/yolov2-tiny-voc.weights).

### Quick Start

1. Installation

    ```bash
    $ git clone https://github.com/wns349/tensorflow-yolo

    # Install requirements. Hopefully, `pip` takes care of it all
    $ pip install -r requirements.txt
    ```


2. Download pre-trained weights file

    Download it from [here](https://pjreddie.com/media/files/yolov2-tiny-voc.weights) and keep it under `/bin` directory.

3. Run prediction

    ```bash
    $ python ./main.py --image ./img/sample_dog.jpg
    ```

    After a successful inference, a resulting image with bounding boxes is saved as `./img/sample_dog_out.jpg`.

    Checkout `main.py` to see a list of available command-line arguments.


### TODO
- Full yolo network
- Train custom objects for detection
- Yolo v3

##### References
I found the following projects/websites to be very helpful. Many thanks!
- [Darknet](https://github.com/pjreddie/darknet/)
- [YAD2K](https://github.com/allanzelener/YAD2K)
- [Darkflow](https://github.com/thtrieu/darkflow/)
- [MLBLR#yolov2](https://mlblr.com/includes/mlai/index.html#yolov2)
- [Implementing YOLO v3 in Tensorflow(TF-Slim)](https://github.com/mystic123/tensorflow-yolo-v3)