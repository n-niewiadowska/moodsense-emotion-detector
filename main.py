import argparse
from preprocessing import preprocessing
from training import train_model
from model_plots import make_plots
from detector import run_detector


def main():
  DATASET = "./emotions-dataset"

  parser = argparse.ArgumentParser(description="MoodSense - Emotion Detector")
  parser.add_argument("-t", "--train", action="store_true", help="Train the model")
  parser.add_argument("-r", "--run", action="store_true", help="Run the detector")
  args = parser.parse_args()

  print("||   (✿◠‿◠) Welcome to MOODSENSE!")
  print("|| Your small Emotion Detector ;3")
  print("||")

  if args.train:
    print("||   Starting data preprocessing...")
    train, validation = preprocessing(DATASET)
    print("||   Images prepared successfully!")

    print("||   Training the model...")
    model, history = train_model(train, validation)
    make_plots(model, history, validation)
    print("||   Detection model is trained and saved as detection_model.keras!")
    print("||   You can find model plots in directory /plots.")
  elif args.run:
    print("||   MoodSense uses a camera to detect your emotions.")
    use_camera = input("||   Do you want to turn on your camera? [y|n]   ")
    if use_camera == "y":
      run_detector()
    else:
      print("||   Okay then. Have a nice day! ;3")
      return
  else:
    print("||   You can run the program with following options:")
    print("||   ✿ -t, --train  - performs data preprocessing, trains the detection model and shows important statistics")
    print("||   ✿ -r, --run    - runs the main detection program")
    print("||")
    print("||   Let's go!")


if __name__ == "__main__":
  main()