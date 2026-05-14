import cv2
import time
from gtts import gTTS
from playsound import playsound
import os
from googletrans import Translator
from ultralytics import YOLO
import threading


DETECTION_DELAY = 1     
COOLDOWN = 2            
MODEL_PATH = "yolov8m.pt"  
CAM_INDEX = 0          


model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAM_INDEX)
translator = Translator()

detected_label = None
start_time = None
is_speaking = False

print("Camera ON... detecting object")


def speak(text, lang, filename):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# Run speech in a separate thread
def speak_thread(msgs):
    global is_speaking
    for msg, lang, file in msgs:
        speak(msg, lang, file)
        time.sleep(0.5)  # small pause between languages
    time.sleep(COOLDOWN)  # wait before resuming detection
    is_speaking = False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    label_now = None

    if not is_speaking:
        # Object detection
        results = model(frame, conf=0.5)

        for r in results:
            if len(r.boxes) > 0:
                box = r.boxes[0]
                cls = int(box.cls[0])
                label_now = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_now, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detection", frame)

   
    if not is_speaking:
        if label_now:
            if detected_label == label_now:
                if time.time() - start_time >= DETECTION_DELAY:
                    is_speaking = True

                    try:
                        label_hi = translator.translate(label_now, dest='hi').text
                        label_ta = translator.translate(label_now, dest='ta').text
                    except:
                        label_hi = label_now
                        label_ta = label_now

                    msg_en = f"{label_now} is in front of you"
                    msg_hi = f"आपके सामने {label_hi} है"
                    msg_ta = f"உங்கள் முன்னால் {label_ta} உள்ளது"

                    # Print messages
                    print(msg_en)
                    print(msg_hi)
                    print(msg_ta)

                    # Start speech in a thread
                    msgs = [
                        (msg_en, "en", "en.mp3"),
                        (msg_hi, "hi", "hi.mp3"),
                        (msg_ta, "ta", "ta.mp3")
                    ]
                    threading.Thread(target=speak_thread, args=(msgs,)).start()

                    # Reset for next detection
                    detected_label = None
                    start_time = None
            else:
                detected_label = label_now
                start_time = time.time()
        else:
            detected_label = None
            start_time = None


    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()