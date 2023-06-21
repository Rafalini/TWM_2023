import cv2
from object_detection import ObjectDetection
import math
import glob
import os


def check_lane_crossing(curr_point, prev_point, lines):
    for line in lines:
        lane_start_x, lane_start_y, lane_end_x, lane_end_y = line

        prev_pos_start = (lane_end_x - lane_start_x) * (prev_point[1] - lane_start_y) - (lane_end_y - lane_start_y) * (prev_point[0] - lane_start_x)
        prev_pos_end = (lane_end_x - lane_start_x) * (prev_point[1] - lane_end_y) - (lane_end_y - lane_start_y) * (prev_point[0] - lane_end_x)
        curr_pos_start = (lane_end_x - lane_start_x) * (curr_point[1] - lane_start_y) - (lane_end_y - lane_start_y) * (curr_point[0] - lane_start_x)
        curr_pos_end = (lane_end_x - lane_start_x) * (curr_point[1] - lane_end_y) - (lane_end_y - lane_start_y) * (curr_point[0] - lane_end_x)

        if prev_pos_start * prev_pos_end <= 0 and curr_pos_start * curr_pos_end > 0:
            return True
        elif prev_pos_start * prev_pos_end >= 0 and curr_pos_start * curr_pos_end < 0:
            return True
    return False


def draw_lanes(lanes):
    for lane in lanes:
        cv2.line(frame, (lane[0], lane[1]), (lane[2], lane[3]), (255, 0, 0), 2)
        cv2.line(frame, (lane[0], lane[1]), (lane[2], lane[3]), (255, 0, 0), 2)
        cv2.line(frame, (lane[0], lane[1]), (lane[2], lane[3]), (255, 0, 0), 2)
        cv2.line(frame, (lane[0], lane[1]), (lane[2], lane[3]), (255, 0, 0), 2)


def draw_limits(limitsL1, limitsL2, limitsR1, limitsR2):
    # left side of the road
    cv2.line(frame, (limitsL1[0], limitsL1[1]),(limitsL1[2], limitsL1[3]), (0,0,255), 2)
    cv2.line(frame, (limitsL2[0], limitsL2[1]), (limitsL2[2], limitsL2[3]), (0, 0, 255), 2)

    # right side of the road
    cv2.line(frame, (limitsR1[0], limitsR1[1]), (limitsR1[2], limitsR1[3]), (0, 255, 0), 2)
    cv2.line(frame, (limitsR2[0], limitsR2[1]), (limitsR2[2], limitsR2[3]), (0, 255, 0), 2)


# Initialize Object Detection
od = ObjectDetection()
#path = glob.glob(os.path.join("./data/examples",'MVI_40191','*.jpg'))
path = glob.glob(os.path.join("./data/examples",'MVI_20034','*.jpg'))
path.sort()

# Initialize frame count
frameCount = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

#limitsL2 = [0, 210, 470, 210] #MVI_40191
#limitsL1 = [300, 90, 520, 90] #MVI_40191
#limitsR1 = [550, 210, 1000, 210] #MVI_40191
#limitsR2 = [540, 90, 750, 90] #MVI_40191


limitsL2 = [0, 370, 530, 370] #MVI_20034
limitsL1 = [315, 100, 530, 100] #MVI_20034
limitsR1 = [560, 370, 1000, 370] #MVI_20034
limitsR2 = [560, 100, 765, 100] #MVI_20034
totalCountL = []
totalCountR = []

enter_times = {}
leave_times = {}
speeds = {}

#lanes
lanes = [[85, 540, 440, 90], [318, 540, 485, 90], [865, 540, 590, 90], [960, 390, 640, 90]] #MVI_20034
#lanes = [[0, 360, 407, 70], [0, 495, 443, 70], [175, 540, 478, 70], [860, 540, 580, 70], [960, 410, 620, 70], [960, 280, 658 , 70]] #MVI_40191
crossLaneCount = []

for file in path:
    frame = cv2.imread(file)
    frameCount += 1

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N°", frameCount, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if frameCount <= 2:
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

                    if check_lane_crossing(pt, pt2, lanes):
                        crossLaneCount.append(object_id)
                        cv2.circle(frame, pt, 20, (0, 0, 255), 2)

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

    draw_limits(limitsL1, limitsL2, limitsR1, limitsR2)
    draw_lanes(lanes)

    for object_id, pt in tracking_objects.items():
        #cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        #cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        if limitsL1[0] < pt[0] < limitsL1[2] and limitsL1[1] - 10 < pt[1] < limitsL1[1] + 10:
            if object_id not in enter_times:
                enter_times[object_id] = frameCount

        elif limitsR1[0] < pt[0] < limitsR1[2] and limitsR1[1] - 10 < pt[1] < limitsR1[1] + 10:
            if totalCountR.count(object_id) == 0:
                totalCountR.append(object_id)
                enter_times[object_id] = frameCount

        elif limitsL2[0] < pt[0] < limitsL2[2] and limitsL2[1] - 10 < pt[1] < limitsL2[1] + 10:
            if totalCountL.count(object_id) == 0:
                totalCountL.append(object_id)
            if object_id in enter_times and object_id not in leave_times:
                leave_times[object_id] = frameCount
                speeds[object_id] = (leave_times[object_id]-enter_times[object_id])/25.0
                cv2.putText(frame, str(f'{speeds[object_id]:.2f} s'), (pt[0], pt[1]-10), 1, 2, (0, 0, 255), 2)

        elif limitsR2[0] < pt[0] < limitsR2[2] and limitsR2[1] - 10 < pt[1] < limitsR2[1] + 10:
            if object_id in enter_times and object_id not in leave_times:
                leave_times[object_id] = frameCount
                speeds[object_id] = (leave_times[object_id]-enter_times[object_id])/25.0
                cv2.putText(frame, str(f'{speeds[object_id]:.2f} s'), (pt[0], pt[1]), 1, 2, (0, 0, 255), 2)


    print("FRAME: " + str(frameCount))

    print("TRACKING OBJECTS")
    print(tracking_objects)

    print("SPEEDS")
    print(speeds)

    # print Count information on the frame
    cv2.putText(frame, str(f'Count right: {len(totalCountR)}'), (700, 50), 3, 1, (255, 255, 255), 1)
    cv2.putText(frame, str(f'Count left: {len(totalCountL)}'), (30, 50), 3, 1, (255, 255, 255), 1)
    cv2.putText(frame, str(f'Lanes crossed: {len(crossLaneCount)}'), (400, 500), 3, 1, (255, 255, 255), 1)
    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
