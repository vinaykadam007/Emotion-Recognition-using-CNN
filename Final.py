import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import dlib
from math import hypot

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cntr = 0
cntl = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0) #http://192.168.43.69:4747/mjpegfeed

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # hor_line = cv2.line(test_img, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(test_img, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(test_img, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = test_img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    cv2.putText(test_img, str(left_side_white), (50,100),font,2,(0,0,255),3)
    cv2.putText(test_img, str(right_side_white), (50,100),font,2,(0,0,255),3)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


while True:
    ret,test_img=cap.read()
    
    new_frame = np.zeros((500, 500, 3), np.uint8)# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    rt, thresh = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
    
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    faces = detector(gray_img)
    for face in faces:
    #     x1 = face.left()
    #     y1 = face.top()
    #     x2 = face.right()
    #     y2 = face.bottom()
    # #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray_img, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

    #     for n in range(0, 68):
    #         x = landmarks.part(n).x
    #         y = landmarks.part(n).y
    #         cv2.circle(test_img, (x, y), 4, (255, 0, 0), -1)
        if blinking_ratio > 5.0: #5.7
            cv2.putText(test_img, "BLINKING", (50, 150), font, 7, (255, 0, 0))

            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            if gaze_ratio <= 1.5:
                cntr = cntr + 1
                cv2.putText(test_img, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                new_frame[:] = (0, 0, 255)
            elif 2 < gaze_ratio < 2.7:
                cv2.putText(test_img, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
            else:
                cntl = cntl + 1
                new_frame[:] = (255, 0, 0)
                cv2.putText(test_img, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)

    cv2.putText(test_img, "RIGHT Blink Counter:"+str(cntr), (50, 100), font, 1, (0, 40, 255), 2, True)
    cv2.putText(test_img, "LEFT Blink Counter:"+str(cntl), (50, 150), font, 1, (0, 40, 255), 2, True)
    
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,66,225),thickness=4)
        roi_gray=gray_img[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        # print(img_pixels)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        
        predictions = model.predict(img_pixels)
        print(predictions)
        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('anger', 'disgust', 'fearful', 'happiness', 'sadness', 'surprise', 'contempt')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    
    
    resized_img = cv2.resize(test_img, (1000, 700))
    
    cv2.imshow('Facial emotion analysis ',resized_img)
    cv2.imshow('Threshold analysis ',thresh)




    if cv2.waitKey(10) == ord('e'):
        break

cap.release()
cv2.destroyAllWindows