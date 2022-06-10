import cv2
import mediapipe as mp
import PIL.Image, PIL.ImageTk
import tkinter
from enum import Enum
from google.protobuf.json_format import MessageToDict

class Gestures(Enum):
    GESTURE_NONE = 0,
    GESTURE_1 = 1,
    GESTURE_2 = 2, 
    GESTURE_3 = 3,
    GESTURE_4 = 4,
    GESTURE_5 = 5

class ActiveHand(Enum):
    HAND_LEFT = 0,
    HAND_RIGHT = 1,
    HAND_BOTH = 2

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
        hands = []
        if not results.multi_hand_landmarks:
            return None, None, None, None
        for idx, hand in enumerate(results.multi_handedness):
            hand = MessageToDict(hand)
            hands.append(hand)
        labels = [hand["classification"][0]["label"] for hand in hands]
        for i in range(len(results.multi_handedness)):
        #if results.multi_hand_landmarks:
            #for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[i]
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))

        return photo, results, num_hands, labels

    def get_features(self, image):
        return self.hands.process(image)

    def classification(self, results, num_hands):
        if not results.multi_hand_landmarks:
            return None

        gestures = []
        hands = []

        for i in range(num_hands):
            firstFingerOpen, secondFingerOpen, thirdFingerOpen, fourthFingerOpen, fifthFingerOpen = self.get_finger_results(results.multi_hand_landmarks[i])
            gesture = self.classify(firstFingerOpen, secondFingerOpen, thirdFingerOpen, fourthFingerOpen, fifthFingerOpen)
            gestures.append(gesture)

        return gestures, hands

    def classify(self, firstFingerOpen, secondFingerOpen, thirdFingerOpen, fourthFingerOpen, fifthFingerOpen):
        #print(f"First finger: {firstFingerOpen}, Second finger: {secondFingerOpen}, Third finger: {thirdFingerOpen}, Fourth finger: {fourthFingerOpen}, Fifth finger: {fifthFingerOpen}")
        if secondFingerOpen and thirdFingerOpen and fourthFingerOpen and fifthFingerOpen:
            return Gestures.GESTURE_5
        if secondFingerOpen and thirdFingerOpen and fourthFingerOpen:
            return Gestures.GESTURE_4
        if secondFingerOpen and thirdFingerOpen:
            return Gestures.GESTURE_3
        if secondFingerOpen:
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

        landmarks = results
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
        