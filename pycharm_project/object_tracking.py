import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import glob
import os


# Initialize Object Detection
od = ObjectDetection()


path = glob.glob(os.path.join("../dane/Insight-MVT_Annotation_Train",'MVI_40191','*.jpg'))

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

limitsL = [0, 210, 470, 210]
limitsR = [550, 210, 1000, 210]
totalCountL = []
totalCountR = []

for file in path:
    frame = cv2.imread(file)
    count += 1

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    # left side of the road
    cv2.line(frame, (limitsL[0], limitsL[1]),(limitsL[2], limitsL[3]), (0,0,255), 5)
    # right side of the road
    cv2.line(frame, (limitsR[0], limitsR[1]), (limitsR[2], limitsR[3]), (0, 255, 0), 5)

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        if limitsL[0] < pt[0] < limitsL[2] and limitsL[1] - 30 < pt[1] < limitsL[1] + 30:
            if totalCountL.count(object_id) == 0:
                totalCountL.append(object_id)

        if limitsR[0] < pt[0] < limitsR[2] and limitsR[1] - 30 < pt[1] < limitsR[1] + 30:
            if totalCountR.count(object_id) == 0:
                totalCountR.append(object_id)

    print("Tracking objects")
    print(tracking_objects)

    print("CUR FRAME NEW PTS")
    print(center_points_cur_frame)

    # print Count information on the frame
    cv2.putText(frame, str(f'Count right: {len(totalCountR)}'), (700, 50), 3, 1, (255, 255, 255), 1)
    cv2.putText(frame, str(f'Count left: {len(totalCountL)}'), (30, 50), 3, 1, (255, 255, 255), 1)
    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
