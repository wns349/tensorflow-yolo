# TensorFlow-Yolo
Python + TensorFlow implementation of YOLO (You Only Look Once).

### Requirements
Refer to `requirements.txt`


### YOLO Networks

- Yolo V2: [config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg), [weights](https://pjreddie.com/media/files/yolov2.weights)
- Yolo V3: [config](), [weights]()

### Quick Start

1. Installation

    ```bash
    $ git clone https://github.com/wns349/tensorflow-yolo

    # Install requirements. Hopefully, `pip` takes care of it all
    $ pip install -r requirements.txt
    ```


2. Download pre-trained weights file

3. Update config file in `config/`
    - `yolo_2.ini`: Use this for Yolo V2
    - `yolo_3.ini`: Use this for Yolo V3

4. Run `launcher.py`
    ```bash
    $ python launcher.py --config PAHT_TO_CONFIG_FILE --mode (TRAIN|TEST|ANCHOR)
    ```

    - `config`: path to configuration file
    - `mode`: _train_, _anchor_, or _test_


### TODO
- Yolo V3

##### References
I found the following projects/websites to be very helpful. Many thanks!
- [Darknet](https://github.com/pjreddie/darknet/)
- [YAD2K](https://github.com/allanzelener/YAD2K)
- [Darkflow](https://github.com/thtrieu/darkflow/)
- [MLBLR#yolov2](https://mlblr.com/includes/mlai/index.html#yolov2)
- [Implementing YOLO v3 in Tensorflow(TF-Slim)](https://github.com/mystic123/tensorflow-yolo-v3)
