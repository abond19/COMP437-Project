import tkinter
import PIL.Image, PIL.ImageTk
import speech_recognition as sr
from VideoCapture import VideoCapture
import mediapipe as mp
from tensorflow.keras.models import load_model
import cv2
import sys
from hand_gestures import HandGestures, Gestures
import os
from AudioCapture import AudioCapture
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class GUI():
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.image = None
        self.window.geometry("1000x1000")
        self.hand_gestures = HandGestures()
        self.recently_took_action = False
        self.update_count = 0
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands   
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.applications = []
        
        #self.model = load_model("mp_hand_gesture")
        self.es = Elasticsearch(['http://localhost:9200'])

        self.setup_applications()

        self.r = sr.Recognizer()
        self.mic = sr.Microphone()

        self.vid = VideoCapture(video_source)
        with self.mic as source:
            self.r.adjust_for_ambient_noise(source)
        self.r.listen_in_background(self.mic, self.callback)

        self.delay = 5
        self.update()
        self.window.mainloop()
    
    def exit(self):
        sys.exit()
    
    def setup_applications(self):
        d = '/Applications'
        records = []
        apps = os.listdir(d)
        for app in apps:
            if os.path.isdir(app):
                continue
            record = {}
            record["application"] = app.split('.app')[0]
            record['voice_command'] = 'open ' + app.split('.app')[0]
            record['sys_command'] = 'open ' + d +'/%s' %app.replace(' ','\ ')
            records.append(record)
        
        self.applications = records

    def search(self, query):
        #q = "open Telegram"
        if query.lower() == "exit application":
            print("Exiting")
            os._exit(0)
        
        words = query.split(" ")
        if words[0].lower() == "open" and len(words) >= 2:
            for record in self.applications:
                if query.lower() == record['voice_command'].lower():
                    return record['sys_command']

        return None
                    
    
    def callback(self, recognizer, audio):
        print("In callback")
        try:
            recognized = recognizer.recognize_google(audio)
            print("Recognized: " + recognized)
            command = self.search(recognized)
            if command == None:
                return
            print(f"Command: {command}")
            os.system(command)
        except Exception as e:
            print(e.with_traceback)
            return
    
    def update(self):
        #self.audio_capture.recognize()
        ret, image = self.vid.get_frame()
        self.update_count += 1
        if ret:
            image.flags.writeable = False
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            photo, results, num_hands, labels = self.hand_gestures.processImage(image)
            image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
            if results == None:
                self.recently_took_action = False
                if self.image is None:
                    self.image = tkinter.Label(self.window, image=image, compound=tkinter.BOTTOM)
                    self.image.place(x=0, y=0)
                else:
                    self.image.configure(image=image)
                    self.image.image = image
                self.window.after(self.delay, self.update)
                return
            gestures = self.hand_gestures.classification(results, num_hands)

            gesture, label = self.get_first_nonzero(gestures, labels)

            self.apply_action(gesture, label)

            if self.image is None:
                self.image = tkinter.Label(self.window, image=photo, text=f"Gesture: {gesture}", compound=tkinter.BOTTOM)
                self.image.place(x=0, y=0)
            else:
                self.image.configure(image=photo, text=f"Gesture: {gesture}")
                self.image.image = photo
                #cv2.imshow("Hands", cv2.flip(image, 1))

        self.window.after(self.delay, self.update)

    def apply_action(self, gesture, hand):
        print(f"Gesture: {gesture}, Hand: {hand}")
        gesture = gesture[0]
        if gesture == Gestures.GESTURE_NONE or gesture is None:
            self.recently_took_action = False
        if gesture == Gestures.GESTURE_5 and not self.recently_took_action and hand == "Right":
            os.system("osascript -e 'tell application \"System Events\" to key code 124 using control down'")
            self.recently_took_action = True
        if gesture == Gestures.GESTURE_5 and not self.recently_took_action and hand == "Left":
            os.system("osascript -e 'tell application \"System Events\" to key code 123 using control down'")
            self.recently_took_action = True
        if (gesture == Gestures.GESTURE_2 or gesture == Gestures.GESTURE_1) and not self.recently_took_action:
            os.system("osascript -e 'tell application \"System Events\" to key code 24 using command down'")
            self.recently_took_action = True
        if gesture == Gestures.GESTURE_3 and not self.recently_took_action:
            os.system("osascript -e 'tell application \"System Events\" to key code 27 using command down'")
            self.recently_took_action = True

    def get_first_nonzero(self, gestures, labels):
        for i in range(len(gestures)):
            if gestures[i] != Gestures.GESTURE_NONE:
                return gestures[i], labels[i]
        else:
            return Gestures.GESTURE_NONE, "None"