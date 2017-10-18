import cv2
from scipy import misc
import numpy as np
import glob, os
import sys
import argparse
import shutil


def move_file(from_images_dir, to_images_dir, filename, check_only):
    filename = os.path.join(from_images_dir, filename)
    if check_only == 1:
        print(filename)
        return
    try:
        shutil.move(filename, to_images_dir)
        print('%s has been moved' % filename)
    except OSError:
        print(OSError)


def hero_pixel_number(filename):
    img = cv2.imread(filename)
    blue = img[:,:,0]

    if np.any(blue == 255):
        s = np.sum(blue==255)
        return s
    return 0

def check_hero_pixel(filename, min_pixels, max_pixels):
    s = hero_pixel_number(filename)
    if s >= min_pixels:
        if max_pixels == -1 or s <= max_pixels:
            return True
    return False

def get_image_file_name(images_dir, filename):
    search_filename = filename.replace('_mask', '*')
#            print(search_filename)
    file_set = glob.glob(images_dir + '/' + search_filename + '.jpeg')
    for name in file_set:
#                print(name)
#                print(len(file_set))
        if len(file_set) == 1:   
            return os.path.splitext(os.path.basename(name))[0] 
    return ''

if __name__ == '__main__':
    max_pixels = 0
    parser = argparse.ArgumentParser()

    parser.add_argument('from_path', help='The image path to filter and move from')
    parser.add_argument('to_path', help='The image path to filter and move to')
    parser.add_argument('-minpixel', "--minpixel", default=0, type=int)
    parser.add_argument('-maxpixel', "--maxpixel", default=-1, type=int)
    parser.add_argument('-check', "--check", default=1, type=int)
    args = parser.parse_args()
#    print(args)
    base_from_path = args.from_path
    print(base_from_path)
    base_to_path = args.to_path
    print(base_to_path)
    max_pixels = args.maxpixel
    print(max_pixels)
    min_pixels = args.minpixel
    print(min_pixels)
    check_only = args.check
    from_mask_dir = base_from_path + '/masks'
    from_images_dir =  base_from_path + '/images'
    to_mask_dir = base_to_path + '/masks'
    to_images_dir =  base_to_path + '/images'
    if not os.path.exists(to_mask_dir):
        os.makedirs(to_mask_dir)
    if not os.path.exists(to_images_dir):
        os.makedirs(to_images_dir)
#    print(mask_dir)
#    print(images_dir)

    moved_file_count = 0
    files = glob.glob(os.path.join(from_mask_dir, '*.png'))
    if len(files) == 0:
        print('No files found in ', from_mask_dir)
    else:
        print('found %d files in %s' % (len(files), from_mask_dir))
    for f in files:
        valid_moved = check_hero_pixel(f, min_pixels, max_pixels)
        if valid_moved:
#            print(f)
            filename = os.path.splitext(os.path.basename(f))[0]
#            print(filename)
            
            img_filename = get_image_file_name(from_images_dir, filename)
            if img_filename != '':
                move_file(from_mask_dir, to_mask_dir, filename + '.png', check_only)
#                    print(image_filename)
                move_file(from_images_dir, to_images_dir, img_filename + '.jpeg', check_only)
                moved_file_count += 1

    print('Total moved file number = %d for images and mask each.' % moved_file_count)

