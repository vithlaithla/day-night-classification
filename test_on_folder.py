# USAGE
# python test_model.py --model model/day_night.hd5 --image test_images/d2.jpg

# import the necessary packages
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os 
from os import listdir
from os.path import isfile, join
os.environ['CUDA_VISIBLE_DEVICES'] ="0"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-dir", "--folder", required=True)
args = vars(ap.parse_args())

days = 0
nights = 0
ud = 0
# load the image
images = [f for f in listdir(args["folder"]) if isfile(join(args["folder"], f))]
print(images)
for image in images:
  image = cv2.imread(join(args["folder"], image))
#   image = cv2.imread(args["image"])
  orig = image.copy()

  # pre-process the image for classification
  image = cv2.resize(image, (28, 28))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)

  # load the trained convolutional neural network
  print("[INFO] loading network...")
  model = load_model(args["model"])

  # classify the input image
  (night, day) = model.predict(image)[0]
  
  if day > night:
    days += 1
  elif day < night:
    nights += 1
  else:
    ud += 1

print("Nights", nights)
print("Days", days)
print("UD", ud)
