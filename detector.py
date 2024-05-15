import cv2, os, random, time, multiprocessing
from keras.models import load_model
from playsound import playsound
import numpy as np
from PIL import Image


def run_detector():
  model = load_model("detection_model_3.keras")
  cv2.ocl.setUseOpenCL(False)
  labels = { 0: "anger", 1: "neutral", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise" }
  
  webcam = cv2.VideoCapture(0)
  print("||   Started the detection program!")
  print("||   Press Q when you want to exit.")

  detected_emotions = []
  last_action_time = time.time()

  while True:
    ret, frame = webcam.read()
    if not ret:
      break

    facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (225, 79, 215), 2)
      roi_gray = gray[y:y + h, x:x + w]
      colored_image = crop_and_color(roi_gray)
      prediction = model.predict(colored_image, verbose=0)
      maxindex = int(np.argmax(prediction))
      
      cv2.putText(frame, labels[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      detected_emotions.append(labels[maxindex])

    if time.time() - last_action_time > 30:
      most_common_emotion = max(set(detected_emotions), key=detected_emotions.count)
      perform_action_for_emotion(most_common_emotion)
      detected_emotions = []
      last_action_time = time.time()

    cv2.imshow("MoodSense", cv2.resize(frame, (600, 400), interpolation=cv2.INTER_CUBIC))
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  webcam.release()
  cv2.destroyAllWindows()
  print("||   The app is closing. Goodbye!")


def crop_and_color(image):
  resized = cv2.resize(image, (48, 48))
  expanded_1 = np.expand_dims(resized, axis=0)
  expanded_2 = np.expand_dims(expanded_1, axis=-1)
  colored_img = cv2.cvtColor(expanded_2[0], cv2.COLOR_GRAY2BGR)
  img_expanded = np.expand_dims(colored_img, axis=0)

  return img_expanded

def play_media(media_folder):
  actions_folder = "./actions"
  media_files = [ f for f in os.listdir(os.path.join(actions_folder, media_folder)) ]

  if not media_files:
    print("Some music or cute cat videos will definitely help you! <3")
    return
  
  selected_file = random.choice(media_files)
  file_path = os.path.join(actions_folder, media_folder, selected_file)

  if selected_file.endswith(".mp3"):
    print("Listen to some relaxing music!")
    p = multiprocessing.Process(target=playsound, args=(file_path,))
    p.start()
    time.sleep(20)
    p.terminate()
    print("♫⋆｡♪ ₊˚♬ ﾟ.")
  elif selected_file.endswith((".jpg", ".png")):
    print("Look at this cutie!!")
    img = Image.open(file_path)
    img.show("Cute animal on the rescue!")

def perform_action_for_emotion(emotion):
  print(f"_|   You clearly feel a lot of {emotion} now...")

  match emotion:
    case "anger":
      print(" (⩺_⩹)   I'm sorry to see that your angry. Close your eyes and take a few deep breaths!")
      play_media("music")
    case "neutral":
      print(" ಠ ಠ   I won't disturb you then. Have a nice day!")
    case "disgust":
      print(" (⊙_☉)   Take care, sweetheart. Here's a cute animal to help you!")
      play_media("pictures")
    case "fear":
      print(" ヽ(O_O )ﾉ   Hey... let's breathe together! Everything is fine!")
      print("... or is it?")
    case "happiness":
      print(" ( ͡^ ͜ʖ ͡^ )   It's great to see you so happy! Have the most wonderful day!")
      play_media("pictures")
    case "sadness":
      print(" (´°ω°`)   I'm so sorry :( Everything will get better though!")
      print(" I highly recommend some deep stretching and meditation. They'll help both your mind and body.")
      if random.randint(0, 1) == 1:
        play_media("music")
      else:
        play_media("pictures")
    case "surprise":
      print(" ( ◐ o ◑ )   Whoah, that must be shocking!")
      print("Want some popcorn?")
      img = Image.open(os.path.join("./actions", "popcorn.jpg"))
      img.show("Popcorn!")
    case _:
      print(" ( *∵* )   Keep up the good work!")
