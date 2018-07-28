import argparse
from net import yolov2
from net import yolov3


def _predict(args):
    yolo = None
    if args.network == "v2":
        yolo = yolov2.YoloV2()
    elif args.network == "v3":
        yolo = yolov3.YoloV3()
    else:
        raise ValueError("Unknown network version")

    # anchors and class names
    with open(args.names, "r") as f:
        names = [n.lstrip().rstrip() for n in f.readlines()]
    with open(args.anchors, "r") as f:
        anchors = [float(t) for t in f.readline().strip().split(",")]

    yolo.initialize({
        "anchors": anchors,
        "names": names
    }, False)
    yolo.predict(args.img_dir,
                 args.out_dir,
                 float(args.threshold),
                 float(args.iou_threshold),
                 args.batch_size,
                 args.weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", dest="img_dir", help="Path to test image directory")
    parser.add_argument("--out_dir", dest="out_dir", help="Path to output directory")
    parser.add_argument("--weights", dest="weights", help="Path to pretrained yolo weight file")
    parser.add_argument("--names", dest="names", help="Path to class name file")
    parser.add_argument("--anchors", dest="anchors", help="Path to anchor file")
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=1)
    parser.add_argument("--threshold", dest="threshold", help="Object detection threshold", default=0.5)
    parser.add_argument("--iou_threshold", dest="iou_threshold", help="NMS threshold", default=0.5)
    parser.add_argument("--network", dest="network", help="Network version (v2 or v3)", default="v2")
    args = parser.parse_args()

    _predict(args)
