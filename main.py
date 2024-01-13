import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import dlib
model = tf.keras.models.load_model("emotion_detection.h5")
face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,img=cap.read()
    img = cv2.flip(img,1)
    height, width, channel = img.shape
    sub_img = img[0:int(height/6),0:int(width)]
    heading = np.ones(sub_img.shape, dtype=np.uint8)*0
    result = cv2.addWeighted(sub_img, 0.82, heading,0.18, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    lable_color = (0, 255, 0)
    lable = "Emotion Detection"
    lable_dimension = cv2.getTextSize(lable,font,font_scale,font_thickness)[0]
    textX = int((result.shape[1] - lable_dimension[0]) / 2)
    textY = int((result.shape[0] + lable_dimension[1]) / 2)
    cv2.putText(result, lable, (textX,textY), font, font_scale, (0,0,0), font_thickness)
    gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image)
    try:
        for (x,y, w,h) in faces:
            cv2.rectangle(img, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness = 2)
            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray=cv2.resize(roi_gray,(48,48))
            shape = predictor(gray_image, dlib.rectangle(x, y, x+w, y+h))
            landmarks = np.zeros((68, 2), dtype=np.int)
            for i in range(0, 68):
                landmarks[i] = (shape.part(i).x, shape.part(i).y)
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_eye_center = left_eye.mean(axis=0).astype("int")
            right_eye_center = right_eye.mean(axis=0).astype("int")
            eye_centers = np.vstack((left_eye_center, right_eye_center))
            gaze_vector = right_eye_center - left_eye_center
            gaze_angle = np.arctan2(gaze_vector[1], gaze_vector[0]) * 180 / np.pi
            image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('You seem Angry.', 'You seem Disgusted.', 'Fear Detected!!', "Yayy, You are not listening.", 'You are not interested.', 'Surprised!!!', 'You are listening.')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(img,emotion_prediction, (int(x),int(y)),font,0.9, lable_color,2)

            x_center = x + int(w/2)
            y_center = y + int(h/2)
            if x_center < int(width/3):
                direction = "Not Attentive"
            elif x_center > int(2*width/3):
                direction = "Not Attentive"
            else:
                direction = "Attentive"
            cv2.putText(img, direction, (int(x), int(y-30)), font, 0.7, lable_color, 2)
    except :
        pass
    img[0:int(height/6),0:int(width)] = result
    cv2.imshow('Emotion Detection', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
