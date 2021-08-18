import cv2
import numpy as np
import math
import argparse
import os

START_POINT = 160
END_POINT = 150
CLASSES = ["car","motorbike","bicycle","truck","van"]
# Define vehicle class
VEHICLE_CLASSES = [0,1,2,3,4]

# get it at https://pjreddie.com/darknet/yolo/
YOLOV3_CFG = 'yolov4-tiny-custom.cfg'
YOLOV3_WEIGHT = 'yolov4-tiny-custom_best.weights'

CONFIDENCE_SETTING = 0.7
YOLOV3_WIDTH = 416
YOLOV3_HEIGHT = 416
MAX_DISTANCE = 10

# NguyenNgocTruong
def parser():
    parser = argparse.ArgumentParser(description="Bộ đếm xe")
    parser.add_argument("--input", type=str, default="input.mp4",
                        help="Đường dẫn video nguồn")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Đường dẫn video đích, nếu không nhập thì sẽ chỉ xuất trên màn hình")
    parser.add_argument("--config_file", default="yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")    
    parser.add_argument("--start_point", default=190, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--end_point", default=120, type=int,
                        help="number of images to be processed at the same time")
    return parser.parse_args()
def check_arguments_errors(args):
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))
# Update ngày 18/8/2021

def get_output_layers(net):
    """
    Get output layers of darknet
    :param net: Model
    :return: output_layers
    """
    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    except:
        print("Can't get output layers")
        return None


def detections_yolo3(net, image, confidence_setting, yolo_w, yolo_h, frame_w, frame_h, classes=None):
    """
    Detect object use yolo3 model
    :param net: model
    :param image: image
    :param confidence_setting: confidence setting
    :param yolo_w: dimension of yolo input
    :param yolo_h: dimension of yolo input
    :param frame_w: actual dimension of frame
    :param frame_h: actual dimension of frame
    :param classes: name of object
    :return:
    """
    img = cv2.resize(image, (yolo_w, yolo_h))
    blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
    scale = 0.00392
    # blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_output = net.forward(get_output_layers(net))

    boxes = []
    class_ids = []
    confidences = []

    for out in layer_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_setting and class_id in VEHICLE_CLASSES:
                # print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
                center_x = int(detection[0] * frame_w)
                center_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # NguyenNgocTruong
    boxes_=[]
    class_ids_=[]
    confidences_=[]
    conf_threshold = 0.5
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        boxes_.append(boxes[i])
        class_ids_.append(class_ids[i])
        confidences_.append(confidences[i])
    return boxes_, class_ids_, confidences_
    # return boxes, class_ids, confidences


def draw_prediction(classes, colors, img, class_id, confidence, x, y, width, height):
    """
    Draw bounding box and put classe text and confidence
    :param classes: name of object
    :param colors: color for object
    :param img: immage
    :param class_id: class_id of this object
    :param confidence: confidence
    :param x: top, left
    :param y: top, left
    :param width: width of bounding box
    :param height: height of bounding box
    :return: None
    """
    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))


def check_location(box_y, box_height, height):
    """
    Check center point of object that passing end line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :param height: height of image
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y < abs(END_POINT):
        return True
    else:
        return False


def check_start_line(box_y, box_height):
    """
    Check center point of object that passing start line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y < START_POINT:
        return True
    else:
        return False

# Hàm dùng để lọc các box mà model đã predict ra, giúp cho khả năng đếm chính xác hơn
def LOC(tmp_list_object):
    list_object=[]
    boxes=[]
    class_ids=[]
    confidences=[]
    trackers=[]
    for i in range(len(tmp_list_object)):
        obj = tmp_list_object[i]
        boxes.append(obj['box'])
        confidences.append(obj['confidence'])
        trackers.append(obj['tracker'])
        class_ids.append(obj['id'])
    conf_threshold = 0.5
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        new_object = {
            'id': class_ids[i],
            'tracker': trackers[i],
            'confidence': confidences[i],
            'box': boxes[i]
        }
        list_object.append(new_object)

    return list_object
# NNT sửa ngày 10/08/2021
    
def counting_vehicle(video_input, video_output, skip_frame=1):
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT)

    # Read first frame
    cap = cv2.VideoCapture(video_input)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, video_format, 25, (width, height))

    # Define tracking object
    list_object = []
    number_frame = 0
    # number_vehicle = 0
    number_vehicle = [0,0,0,0,0]
    while cap.isOpened():
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
        # Tracking old object
        tmp_list_object = LOC(list_object)
        list_object = []

        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box

                draw_prediction(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height)

                obj['tracker'] = tracker
                obj['box'] = box
                if check_location(box_y, box_height, height):
                    # This object passed the end line
                    # number_vehicle += 1
                    number_vehicle[class_id]+=1
                else:
                    list_object.append(obj)
        # cv2.imwrite('resurt/abc'+str(number_frame)+'.jpg',frame)
        if number_frame % skip_frame == 0:
            # Detect object and check new object

            boxes, class_ids, confidences = detections_yolo3(net, frame, CONFIDENCE_SETTING, YOLOV3_WIDTH,
                                                             YOLOV3_HEIGHT, width, height, classes=CLASSES)
            
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    # This object doesnt pass the end line
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    check_new_object = True
                    for tracker in list_object:
                        # Check exist object
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        # Calculate distance between 2 object
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                    if check_new_object and check_start_line(box_y, box_height):
                        # Append new object to list
                        new_tracker = cv2.TrackerKCF_create()
                        new_tracker.init(frame, tuple(map(int,box)))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }
                        list_object.append(new_object)
                        # Draw new object
                        draw_prediction(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)
        
        
        # Draw start line
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (204, 90, 208), 1)
        # Draw end line
        cv2.line(frame, (0,  END_POINT), (width, END_POINT), (255, 0, 0), 2)
        
        # Sửa đổi ngày 10/08/2021
        space = '            '
        cv2.rectangle(frame, (5, 5), (125, 110), (150, 75, 0), -1)
        for i_ in range(5):
            cv2.putText(frame, CLASSES[i_], (10, 20*(i_+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
            cv2.putText(frame, ": {:02d}".format(number_vehicle[i_]), (90, 20*(i_+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
        

        # Show frame
        out.write(frame)
        resized = cv2.resize(frame, (width,height), interpolation = cv2.INTER_AREA)
        cv2.imshow("Counting", resized)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)
    START_POINT = args.start_point
    END_POINT = args.end_point
    YOLOV3_CFG = args.config_file
    YOLOV3_WEIGHT = args.weights
    counting_vehicle(args.input, args.output)