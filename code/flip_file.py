import os
import glob
from scipy import misc
import numpy as np

def flip_and_save_images(img_dir, extension):
  os.chdir(img_dir)
  files = glob.glob("*." + extension)
  for i, file in enumerate(files):
    print(i)
    img = misc.imread(file, flatten=False, mode='RGB')
    flipped_img = np.fliplr(img)
    misc.imsave("flipped" + file, flipped_img)

train_mask_dir = "/home/yangnie/udcity/RoboND-DeepLearning-Project/data/train/masks/"
train_images_dir = "/home/yangnie/udcity/RoboND-DeepLearning-Project/data/train/images/"

flip_and_save_images(train_mask_dir, "png")
flip_and_save_images(train_images_dir, "jpeg")
