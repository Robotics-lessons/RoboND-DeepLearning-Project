import glob
import os
import shutil
import sys
import argparse
import logging
import numpy as np
from scipy import misc
import cv2
from datetime import datetime, date

#no hero = 0 pixels
#hero very far = 1 to 40 pixels
#hero not too close = 40 to 400 pixels
#hero close = > 400 pixels

def contains_hero(filename):
    returnCode = 0
    img = cv2.imread(filename)
    blue = img[:,:,0]

    if np.any(blue == 255):
        s = np.sum(blue==255)
#        print(s)
        if s >= 1 and s < 40:
            returnCode = 1
        elif s >= 40 and s < 400:
            returnCode = 2
        else:
            returnCode = 3 
    return returnCode

def count_type_number(base_path):
    counter = [0, 0, 0, 0]
    files = glob.glob(os.path.join(base_path, '*.png'))
    if len(files) == 0:
        logging.info('No files found in ', base_path)
    else:
        logging.info('found %d files in %s' % (len(files), base_path))
    for f in files:
#        print(f)
        counter[contains_hero(f)] += 1
    logging.info('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    logging.info('no hero = %d' %  counter[0])
    logging.info('hero very far = %d' %  counter[1])
    logging.info('hero not too close = %d' %  counter[2])
    logging.info('hero close = %d' %  counter[3])


if __name__ == '__main__':
    counter = [0, 0, 0, 0]
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='../logs/type_count_' + datetime.now().strftime('model_%Y-%m-%d-%H-%M-%S') + '.log',
                    filemode='w')
    parser = argparse.ArgumentParser()

    parser.add_argument('-path', '--path', default='../data/train/masks/',
                        help='The image path to filter')

    args = parser.parse_args()
    print(args)
    base_path = args.path
    files = glob.glob(os.path.join(base_path, '*.png'))
    if len(files) == 0:
        print('No files found in ', base_path)
    else:
        print('found %d files in %s' % (len(files), base_path))
    for f in files:
#        print(f)
        counter[contains_hero(f)] += 1
 #       pass

    print('no hero (pixel = 0), file number = %d' %  counter[0])
    print('hero very far (pixel > 0 and < 40), file number = %d ' %  counter[1])
    print('hero not too close (pixel >= 40 and < 400), file number = %d' %  counter[2])
    print('hero close (pixel >= 400), file number = %d' %  counter[3])
    count_type_number(base_path)
