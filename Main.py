import pickle
import cv2
import mediapipe as mp
import numpy as np


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.5)


labels_dict = {0: 'C', 1: 'M', 2: 'B', 3: 'S'}
confidence_thresholds = {'C': 0.8, 'M': 0.7, 'B': 0.8, 'S': 0.45}

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process hand landmarks
    results = hands.process(frame_rgb)
    predicted_character = '???'
    confidence = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
        # Pad data_aux if needed to ensure it has 84 features
        while len(data_aux) < 84:
            data_aux.extend([0, 0])
        prediction = model.predict_proba([np.asarray(data_aux)])
        predicted_label = np.argmax(prediction)
        if 0 <= predicted_label < len(labels_dict):
            predicted_character = labels_dict[predicted_label]
            confidence = np.max(prediction)
    x1 = int(min(x_) * W) - 10 if x_ else 0
    y1 = int(min(y_) * H) - 10 if y_ else 0
    x2 = int(max(x_) * W) - 10 if x_ else 0
    y2 = int(max(y_) * H) - 10 if y_ else 0
    box_color = (0, 255, 0)  # Green

    if predicted_character in confidence_thresholds:
        if confidence < confidence_thresholds[predicted_character]:
            predicted_character = '???'
            box_color = (0, 0, 255)  # Red

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3,
                cv2.LINE_AA)


    print("Predicted Character:", predicted_character)
    print("Confidence:", confidence)
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cap.release()
cv2.destroyAllWindows()
