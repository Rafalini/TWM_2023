import cv2
import json
import os
from object_detection import ObjectDetection

image_folder_path = "../data/Insight-MVT_Annotation_Train/MVI_63563"
annotation_folder_path = "Insight-MVT_Annotation_Train/MVI_63563"

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] +bbox1[3], bbox2[1] + bbox2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou

def draw_pred_bounding_boxes(image, bboxes, color):
    for bbox in bboxes:
        x, y, w, h = map(int, bbox)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return image

def draw_gt_bounding_boxes(image, annotations, color):
    for annotation in annotations:
        bbox = annotation["x_y_w_h"]
        x, y, w, h = map(int, bbox)

        cv2.rectangle(image, (x, y), (w, y+h), color, 2)

    return image

od = ObjectDetection()

for image_filename in os.listdir(image_folder_path):
    if image_filename.endswith(".jpg"):

        annotation_filename = os.path.splitext(image_filename)[0] + ".json"
        annotation_path = os.path.join(annotation_folder_path, annotation_filename)

        with open(annotation_path, "r") as f:
            ground_truth = json.load(f)

        image_width, image_height = ground_truth["image_w_h"]

        image_path = os.path.join(image_folder_path, image_filename)
        image = cv2.imread(image_path)

        (class_ids, scores, predicted_bboxes) = od.detect(image)

        # Draw ground truth bounding boxes in green
        image_with_gt_bboxes = draw_gt_bounding_boxes(image.copy(), ground_truth["objects"], (0, 255, 0))

        # Draw predicted bounding boxes in red
        image_with_predicted_bboxes = draw_pred_bounding_boxes(image.copy(), predicted_bboxes, (0, 0, 255))

        # Calculate IoU for each pair of bounding boxes
        for gt_bbox in ground_truth["objects"]:
            for pred_bbox in predicted_bboxes:
                iou = calculate_iou(gt_bbox["x_y_w_h"], pred_bbox)
                if iou > 0:
                    (x, y, w, h) = pred_bbox
                    cx = int((x + x + w) / 2)
                    cy = int((y + y + h) / 2)
                    cv2.putText(image_with_predicted_bboxes, str(f'{iou:.3f} s'), (cx, cy), 1, 2, (0, 0, 255), 2)


        # Display the image with both ground truth and predicted bounding boxes
        cv2.imshow("Image", image_with_gt_bboxes)
        cv2.imshow("Predicted Bounding Boxes", image_with_predicted_bboxes)
        cv2.waitKey(0)


cv2.destroyAllWindows()
