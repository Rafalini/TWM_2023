import cv2
import json
import os

# Folder paths for images and annotations
image_folder_path = "../data/Insight-MVT_Annotation_Train/MVI_63563"
annotation_folder_path = "Insight-MVT_Annotation_Train/MVI_63563"

for image_filename in os.listdir(image_folder_path):
    if image_filename.endswith(".jpg"):

        annotation_filename = os.path.splitext(image_filename)[0] + ".json"
        annotation_path = os.path.join(annotation_folder_path, annotation_filename)

        with open(annotation_path, "r") as f:
            data = json.load(f)

        image_width, image_height = data["image_w_h"]

        image_path = os.path.join(image_folder_path, image_filename)
        image = cv2.imread(image_path)

        for annotation in data["objects"]:
            label = annotation["label"]
            x, y, w, h = map(int, annotation["x_y_w_h"])

            print("x:" + str(x))
            print("y:" + str(y))
            print("w:" + str(w))
            print("h:" + str(h))

            cv2.rectangle(image, (x, y), (w, y+h), (0, 255, 0), 2)
            #cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)  # Wait for a key press before moving to the next image

        cv2.destroyAllWindows()

