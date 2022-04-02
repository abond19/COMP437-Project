import cv2
import mediapipe as mp
import PIL.Image, PIL.ImageTk
import tkinter
from enum import Enum

class Gestures(Enum):
    GESTURE_NONE = 0,
    GESTURE_1 = 1,
    GESTURE_2 = 2, 
    GESTURE_3 = 3,
    GESTURE_4 = 4,
    GESTURE_5 = 5

class HandGestures:
    def __init__(self, model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands   
        self.hands = self.mp_hands.Hands(model_complexity=model_complexity, 
                                        min_detection_confidence=min_detection_confidence, 
                                        min_tracking_confidence=min_tracking_confidence)

        
    def processImage(self, image):
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))

        return photo, results, num_hands

    def get_features(self, image):
        return self.hands.process(image)

    def classification(self, results):
        if not results.multi_hand_landmarks:
            return None

        firstFingerOpen, secondFingerOpen, thirdFingerOpen, fourthFingerOpen, fifthFingerOpen = self.get_finger_results(results)

        gesture = self.classify(firstFingerOpen, secondFingerOpen, thirdFingerOpen, fourthFingerOpen, fifthFingerOpen)
        return gesture

    def classify(self, firstFingerOpen, secondFingerOpen, thirdFingerOpen, fourthFingerOpen, fifthFingerOpen):
        if firstFingerOpen and secondFingerOpen and thirdFingerOpen and fourthFingerOpen and fifthFingerOpen:
            return Gestures.GESTURE_5
        if firstFingerOpen and secondFingerOpen and thirdFingerOpen and fourthFingerOpen:
            return Gestures.GESTURE_4
        if firstFingerOpen and secondFingerOpen and thirdFingerOpen:
            return Gestures.GESTURE_3
        if firstFingerOpen and secondFingerOpen:
            return Gestures.GESTURE_2
        if firstFingerOpen:
            return Gestures.GESTURE_1
        return Gestures.GESTURE_NONE
 
    def get_finger_results(self, results):
        firstFingerOpen = False
        secondFingerOpen = False
        thirdFingerOpen = False
        fourthFingerOpen = False
        fifthFingerOpen = False

        landmarks = results.multi_hand_landmarks[0]
        pseudoFixKeyPoint = landmarks.landmark[2].x
        if landmarks.landmark[3].x < pseudoFixKeyPoint and landmarks.landmark[4].x < pseudoFixKeyPoint:
            firstFingerOpen = True
        
        pseudoFixKeyPoint = landmarks.landmark[6].y
        if landmarks.landmark[7].y < pseudoFixKeyPoint and landmarks.landmark[8].y < pseudoFixKeyPoint:
            secondFingerOpen = True
        
        pseudoFixKeyPoint = landmarks.landmark[10].y
        if landmarks.landmark[11].y < pseudoFixKeyPoint and landmarks.landmark[12].y < pseudoFixKeyPoint:
            thirdFingerOpen = True

        pseudoFixKeyPoint = landmarks.landmark[14].y
        if landmarks.landmark[15].y < pseudoFixKeyPoint and landmarks.landmark[16].y < pseudoFixKeyPoint:
            fourthFingerOpen = True

        pseudoFixKeyPoint = landmarks.landmark[18].y
        if landmarks.landmark[19].y < pseudoFixKeyPoint and landmarks.landmark[20].y < pseudoFixKeyPoint:
            fifthFingerOpen = True

        return firstFingerOpen, secondFingerOpen, thirdFingerOpen, fourthFingerOpen, fifthFingerOpen
        