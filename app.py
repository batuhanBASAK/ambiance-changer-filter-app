import cv2
import numpy as np
import mediapipe as mp
import math


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


width  = cap.get(3)
height = cap.get(4)


states = ['initial', 'waiting']

state = 'initial'
filter_color = 'None'
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:


        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))


            for j, classification in enumerate(results.multi_handedness):
                if i == j:
                    text=classification.classification[0].label

                    if classification.classification[0].label == 'Left': # left hand
                        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x:
                            x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
                            y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                            x = int(x * width)
                            y = int(y * height)
                            x_padding = 50
                            y_padding = 100

                            normal_text_pos = (x+x_padding, y)
                            blue_text_pos = (x+x_padding, y+y_padding)
                            green_text_pos = (x+x_padding, y+2*y_padding)
                            purple_text_pos = (x+x_padding, y+3*y_padding)

                            rect_width = 120
                            rect_height = 50
                            normal_rect_pos = (x+x_padding, y-rect_height//2), (x+x_padding+rect_width, y+rect_height//2)
                            blue_rect_pos = (x+x_padding, y+y_padding-rect_height//2), (x+x_padding+rect_width, y+y_padding+rect_height//2)
                            green_rect_pos = (x+x_padding, y+2*y_padding-rect_height//2), (x+x_padding+rect_width, y+(2*y_padding)+rect_height//2)
                            purple_rect_pos = (x+x_padding, y+3*y_padding-rect_height//2), (x+x_padding+rect_width, y+(3*y_padding)+rect_height//2)




                            cv2.rectangle(frame, normal_rect_pos[0], normal_rect_pos[1], (0, 0, 0), -1)
                            cv2.putText(frame, 'Normal', normal_text_pos, cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            cv2.rectangle(frame, blue_rect_pos[0], blue_rect_pos[1], (0, 0, 0), -1)
                            cv2.putText(frame, 'Blue',   blue_text_pos, cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            cv2.rectangle(frame, green_rect_pos[0], green_rect_pos[1], (0, 0, 0), -1)
                            cv2.putText(frame, 'Green',  green_text_pos, cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            cv2.rectangle(frame, purple_rect_pos[0], purple_rect_pos[1], (0, 0, 0), -1)
                            cv2.putText(frame, 'Red', purple_text_pos, cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)
                            state = 'waiting'
                        else:
                            state = 'initial'

                    else: # right hand
                        x1, y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                        x2, y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
                        distance = math.sqrt(((x1-x2)**2) + ((y1-y2)**2))
                        # print(distance)
                        if classification.classification[0].label == 'Right' and state == 'waiting' and distance < 50.0:
                            middle_point = (x1+x2)//2, (y1+y2)//2 
                            if middle_point[0] >= normal_rect_pos[0][0] and middle_point[0] <= normal_rect_pos[1][0] and middle_point[1] >= normal_rect_pos[0][1] and middle_point[1] <= normal_rect_pos[1][1]:
                                filter_color = 'None'
                            elif middle_point[0] >= blue_rect_pos[0][0] and middle_point[0] <= blue_rect_pos[1][0] and middle_point[1] >= blue_rect_pos[0][1] and middle_point[1] <= blue_rect_pos[1][1]:
                                filter_color = 'blue'
                            elif middle_point[0] >= green_rect_pos[0][0] and middle_point[0] <= green_rect_pos[1][0] and middle_point[1] >= green_rect_pos[0][1] and middle_point[1] <= green_rect_pos[1][1]:
                                filter_color = 'green'
                            elif middle_point[0] >= purple_rect_pos[0][0] and middle_point[0] <= purple_rect_pos[1][0] and middle_point[1] >= purple_rect_pos[0][1] and middle_point[1] <= purple_rect_pos[1][1]:
                                filter_color = 'red'
        

            x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            x = int(x * width)
            y = int(y * height)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)


    if filter_color == 'None':
        cv2.imshow('frame', frame)
    elif filter_color == 'blue':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        h += 120
        frame_blue_hsv = cv2.merge([h,s,v])
        frame_blue = cv2.cvtColor(frame_blue_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame', frame_blue)  
    elif filter_color == 'green':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        h += 60
        frame_green_hsv = cv2.merge([h,s,v])
        frame_green = cv2.cvtColor(frame_green_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame', frame_green)        
    elif filter_color == 'red':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        h -= h
        frame_red_hsv = cv2.merge([h,s,v])
        frame_red = cv2.cvtColor(frame_red_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame', frame_red)  


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
