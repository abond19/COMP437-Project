import tkinter 
import cv2
import mediapipe as mp
import PIL

class HandGestures:
    def __init__(self, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5):
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

        return photo, results