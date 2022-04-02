import tkinter
import PIL.Image, PIL.ImageTk
from VideoCapture import VideoCapture
import mediapipe as mp
from tensorflow.keras.models import load_model
import cv2
import sys
from hand_gestures import HandGestures

class GUI:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.image = None
        self.hand_gestures = HandGestures()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands   
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        #self.model = load_model("mp_hand_gesture")

        self.vid = VideoCapture(video_source)

        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.delay = 5
        self.update()
        self.window.mainloop()
    
    def exit(self):
        sys.exit()
    
    def update(self):
        ret, image = self.vid.get_frame()

        if ret:
            image.flags.writeable = False
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            photo, results = self.hand_gestures.processImage(image)
            if self.image is None:
                self.image = tkinter.Label(self.window, image=photo)
                self.image.place(x=0, y=0)
            else:
                self.image.configure(image=photo)
                self.image.image = photo
                #cv2.imshow("Hands", cv2.flip(image, 1))

        self.window.after(self.delay, self.update)

