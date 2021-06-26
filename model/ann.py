import os
import cv2
import time
import joblib
import mediapipe as mp

import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

single_model = joblib.load('model/models/single_mlp.pkl')
both_model = joblib.load('model/models/both_mlp.pkl')

alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def predict(img) :
    IMAGE_FILES = [img]
    result_dict={}
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.2) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            img = cv2.imread(file)
            height, width, channels = img.shape
            cropped_image = img[height//2-width//2:height//2+width//2, 0:width]
            image = cropped_image # cv2.flip(cropped_image, 1)
            # tfName = f'./temp/crop_{time.time()}.jpg'
            cv2.imwrite(tfName,image)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                    continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for idx,hand_landmarks in enumerate(results.multi_hand_landmarks):
                result_dict['WRIST_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                result_dict['WRIST_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                result_dict['WRIST_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z
                result_dict['THUMB_CMC_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x
                result_dict['THUMB_CMC_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
                result_dict['THUMB_CMC_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z
                result_dict['THUMB_MCP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
                result_dict['THUMB_MCP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
                result_dict['THUMB_MCP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z
                result_dict['THUMB_IP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
                result_dict['THUMB_IP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
                result_dict['THUMB_IP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z
                result_dict['THUMB_TIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                result_dict['THUMB_TIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                result_dict['THUMB_TIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z
                result_dict['INDEX_FINGER_MCP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                result_dict['INDEX_FINGER_MCP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                result_dict['INDEX_FINGER_MCP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
                result_dict['INDEX_FINGER_PIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
                result_dict['INDEX_FINGER_PIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                result_dict['INDEX_FINGER_PIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z
                result_dict['INDEX_FINGER_DIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
                result_dict['INDEX_FINGER_DIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                result_dict['INDEX_FINGER_DIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z
                result_dict['INDEX_FINGER_TIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                result_dict['INDEX_FINGER_TIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                result_dict['INDEX_FINGER_TIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
                result_dict['MIDDLE_FINGER_MCP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                result_dict['MIDDLE_FINGER_MCP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                result_dict['MIDDLE_FINGER_MCP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
                result_dict['MIDDLE_FINGER_PIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
                result_dict['MIDDLE_FINGER_PIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                result_dict['MIDDLE_FINGER_PIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z
                result_dict['MIDDLE_FINGER_DIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
                result_dict['MIDDLE_FINGER_DIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
                result_dict['MIDDLE_FINGER_DIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z
                result_dict['MIDDLE_FINGER_TIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                result_dict['MIDDLE_FINGER_TIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                result_dict['MIDDLE_FINGER_TIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                result_dict['RING_FINGER_MCP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
                result_dict['RING_FINGER_MCP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
                result_dict['RING_FINGER_MCP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z
                result_dict['RING_FINGER_PIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
                result_dict['RING_FINGER_PIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
                result_dict['RING_FINGER_PIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z
                result_dict['RING_FINGER_DIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
                result_dict['RING_FINGER_DIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
                result_dict['RING_FINGER_DIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z
                result_dict['RING_FINGER_TIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
                result_dict['RING_FINGER_TIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                result_dict['RING_FINGER_TIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
                result_dict['PINKY_MCP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
                result_dict['PINKY_MCP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
                result_dict['PINKY_MCP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z
                result_dict['PINKY_PIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x
                result_dict['PINKY_PIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
                result_dict['PINKY_PIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z
                result_dict['PINKY_DIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x
                result_dict['PINKY_DIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
                result_dict['PINKY_DIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z
                result_dict['PINKY_TIP_x'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
                result_dict['PINKY_TIP_y'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                result_dict['PINKY_TIP_z'+'_'+str(idx)] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
            
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    # tfName = f'{time.time()}'
    # cv2.imwrite(f'output/{tfName}.png', cv2.flip(annotated_image, 1))

    new_result = {}
    for i in result_dict:
        new_result[i] = [result_dict[i]]
    new_result_arr = pd.DataFrame(new_result)

    os.remove(img)

    if(len(result_dict)==63):
        response = alphabets[single_model.predict(new_result_arr)[0]]
    elif(len(result_dict)==126):
        response = alphabets[both_model.predict(new_result_arr)[0]]
    else :
        response = ' '

    print(response)

    return response