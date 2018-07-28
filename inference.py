import cv2
import numpy as np


def preprocess(img, size=(416, 416)):
    imsz = cv2.resize(img, size)
    imsz = imsz / 255.  # to make values lie between 0 and 1
    imsz = imsz[:, :, ::-1]  # BGR to RGB
    return imsz


def postprocess(net_out, anchors, threshold, iou_threshold, output_h, output_w):
    results = []
    for out in net_out:
        bounding_boxes = _find_bounding_boxes(out, anchors, threshold, output_h, output_w)
        results.append(_non_maximum_suppression(bounding_boxes, iou_threshold))
    return results


def _non_maximum_suppression(bboxes, iou_threshold):
    if len(bboxes) == 0:
        return []

    bboxes.sort(key=lambda box: box.prob, reverse=True)  # sort by prob(confidence)
    new_boxes = [bboxes[0]]  # add first element
    for i in range(1, len(bboxes)):
        overlapping = False
        for new_box in new_boxes:
            if iou_score(new_box, bboxes[i]) >= iou_threshold:
                overlapping = True
                break
        if not overlapping:
            new_boxes.append(bboxes[i])
    return new_boxes

