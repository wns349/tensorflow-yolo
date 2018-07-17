import argparse
import os

import cv2

from yolov2 import create_full_network, create_tiny_network, load_weights, predict

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

_func = {
    "yolov2-tiny": {
        "builder": create_tiny_network,
        "loader": load_weights
    },
    "yolov2": {
        "builder": create_full_network,
        "loader": load_weights
    }
}


def _main(args):
    print("Hello")
    func = _func[args.network]
    if func is None:
        raise ValueError("Unsupported network.")

    if args.image is None or not os.path.exists(args.image):
        raise ValueError("Invalid image path! Use --image")

    with open(args.labels, "r") as f:
        labels = [l.strip() for l in f.readlines()]
    with open(args.anchors, "r") as f:
        anchors = [float(l.strip()) for l in f.readline().split(",")]

    # helper method to save resulting image
    def _result_callback():
        def _save_resulting_image(result):
            # write result to file
            img = cv2.imread(args.image)
            org_img_h, org_img_w = img.shape[0:2]
            name, ext = os.path.splitext(os.path.abspath(args.image))
            out_path = "{}_out{}".format(name, ext)
            for box in result:
                top_left = box.get_top_left(org_img_h, org_img_w)
                bottom_right = box.get_bottom_right(org_img_h, org_img_w)
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                cv2.rectangle(img, top_left, bottom_right, COLORS[box.class_idx % len(COLORS)], thickness=3)
                label = labels[box.class_idx]
                cv2.putText(img, "{} {:.3f}".format(label, box.prob), (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[box.class_idx % len(COLORS)], thickness=1)
            cv2.imwrite(out_path, img)
            print("Resulting image: {}".format(out_path))
            return out_path

        return _save_resulting_image

    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use this to run on CPU

    params = {
        "scope": "yolo",
        "anchors": anchors,
        "weights": args.weights,
        "labels": labels,
        "threshold": float(args.threshold),
        "iou_threshold": float(args.iou_threshold),
        "builder": func["builder"],
        "loader": func["loader"],
        "test_image": args.image,
        "result_callback": _result_callback()
    }
    predict(params)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", dest="network", help="Yolo network type (yolov2, yolov2-tiny)")
    parser.add_argument("--weights", dest="weights", help="Path to weights file")
    parser.add_argument("--labels", dest="labels", help="Label file")
    parser.add_argument("--anchors", dest="anchors", help="Anchor file")
    parser.add_argument("--threshold", dest="threshold", help="Threshold value", default=0.5)
    parser.add_argument("--iou_threshold", dest="iou_threshold", help="IOU Threshold value", default=0.5)
    parser.add_argument("--image", dest="image", help="Test image")
    parser.add_argument("--mode", dest="mode", help="Mode", default="predict")
    parser.add_argument("--cpu", dest="use_cpu", help="Use CPU", action="store_true", default=False)
    _main(parser.parse_args())
