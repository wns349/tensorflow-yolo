from tensorflow.python.keras.layers import Input
import numpy as np
from keras_net import darknet_yolov2
from tensorflow.python.keras.optimizers import Adam

anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
anchors = np.reshape(anchors, [5, 2])


def test_model(model, img_path):
    class BoundingBox(object):
        def __init__(self, x=0., y=0., w=0., h=0., cx=0, cy=0, class_idx=-1, prob=-1.):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.cx = cx
            self.cy = cy
            self.class_idx = class_idx
            self.prob = prob

        def get_top_left(self, h=1., w=1.):
            return (self.x - self.w / 2.) * w, (self.y - self.h / 2.) * h

        def get_bottom_right(self, h=1., w=1.):
            return (self.x + self.w / 2.) * w, (self.y + self.h / 2.) * h

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def iou_score(box1, box2):
        box1_min, box1_max = box1.get_top_left(), box1.get_bottom_right()
        box1_area = box1.w * box1.h
        box2_min, box2_max = box2.get_top_left(), box2.get_bottom_right()
        box2_area = box2.w * box2.h

        intersect_min = np.maximum(box1_min, box2_min)
        intersect_max = np.minimum(box1_max, box2_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0)
        intersect_area = intersect_wh[0] * intersect_wh[1]
        union_area = np.maximum(box1_area + box2_area - intersect_area, 1e-8)

        return intersect_area / union_area

    import cv2
    # read, preprocess
    org_img = cv2.imread(img_path)
    img = cv2.resize(org_img, (416, 416))
    img = img[:, :, ::-1]
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    out = model.predict(img)

    # postprocess

    threshold = 0.5
    iou_threshold = 0.4
    out = out[0]
    h, w = out.shape[0:2]
    out = np.reshape(out, [h, w, len(anchors), -1])
    bboxes = []
    for cy in range(h):
        for cx in range(w):
            for b in range(len(anchors)):
                prob_obj = sigmoid(out[cy, cx, b, 4])
                prob_classes = softmax(out[cy, cx, b, 5:])
                class_idx = np.argmax(prob_classes)
                class_prob = prob_classes[class_idx]
                p = prob_obj * class_prob
                if p < threshold:
                    continue
                coords = out[cy, cx, b, 0:4]
                bbox = BoundingBox()
                bbox.x = (sigmoid(coords[0]) + cx) / w
                bbox.y = (sigmoid(coords[1]) + cy) / h
                bbox.w = (anchors[b][0] * np.exp(coords[2])) / w
                bbox.h = (anchors[b][1] * np.exp(coords[3])) / h
                bbox.class_idx = class_idx
                bbox.prob = p
                bboxes.append(bbox)

    bboxes.sort(key=lambda box: box.prob, reverse=True)
    new_boxes = [bboxes[0]]
    for i in range(1, len(bboxes)):
        overlapping = False
        for new_box in new_boxes:
            if iou_score(new_box, bboxes[i]) >= iou_threshold:
                overlapping = True
                break
        if not overlapping:
            new_boxes.append(bboxes[i])
    print(new_boxes)
    h, w = org_img.shape[0:2]
    for box in new_boxes:
        tl = np.maximum(box.get_top_left(h, w), 0)
        br = np.maximum(box.get_bottom_right(h, w), 0)
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        org_img = cv2.rectangle(org_img, tl, br, (0, 0, 255), thickness=3)
        print(box.prob)
    cv2.imshow("img", org_img)
    cv2.waitKey(0)

def batch_gen():
    x = np.random.rand(416, 416, 3)
    print(x)

def main():
    input_layer = Input(shape=[416, 416, 3])
    model = darknet_yolov2.build_model(input_layer,
                                       include_top=True,
                                       weights_path="./checkpoints/full-yolov2-coco")

    print(model.summary())

    batch_gen()

    # optimizer = Adam(lr=1e-3)
    # model.compile(optimizer=optimizer, loss=darknet_yolov2.build_loss_fn(anchors))
    # test_model(model, "../img/dog.jpg")


if __name__ == '__main__':
    main()
