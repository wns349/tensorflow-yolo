import argparse
import os

from tiny_yolo import predict

#####
# TODO: extract to configs
VOC_ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
VOC_LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# VOC_WEIGHTS = "./bin/yolov2-tiny-voc.weights"
VOC_WEIGHTS = "./bin/tiny-yolo-voc.weights"


#####


def _main(args):
    print("Hello")
    if args.image is None or not os.path.exists(args.image):
        raise ValueError("Invalid image path! Use --image")

    params = {
        "anchors": VOC_ANCHORS,
        "weights": args.weights,
        "labels": VOC_LABELS,
        "threshold": float(args.threshold),
        "iou_threshold": float(args.iou_threshold)
    }
    out_path = predict(args.image, params)
    print("Done. Detection at {}".format(out_path))


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Use this to run on CPU
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", dest="weights", help="Path to weights file", default=VOC_WEIGHTS)
    parser.add_argument("--threshold", dest="threshold", help="Threshold value", default=0.5)
    parser.add_argument("--iou_threshold", dest="iou_threshold", help="IOU Threshold value", default=0.5)
    parser.add_argument("--image", dest="image", help="Test image")
    _main(parser.parse_args())
