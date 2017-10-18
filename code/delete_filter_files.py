import cv2
from scipy import misc
import numpy as np
import glob, os
import sys
import argparse

def flip_and_save_images(img_dir, filename, check_only):
    flipped_filename = os.path.join(img_dir,'flipped-' + filename)
    filename = os.path.join(img_dir, filename)
#    print(filename)
    print(flipped_filename)
    img = misc.imread(filename, flatten=False, mode='RGB')
    flipped_img = np.fliplr(img)
    if not check_only:
        misc.imsave(flipped_filename, flipped_img)

def delete_file(img_dir, filename, check_only):
    filename = os.path.join(img_dir, filename)
    if check_only == 1:
        print(filename)
        return
    try:
        os.remove(filename)
        print('%s has been deleted' % filename)
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

    parser.add_argument('path', help='The image path to filter')
    parser.add_argument('-minpixel', "--minpixel", default=0, type=int)
    parser.add_argument('-maxpixel', "--maxpixel", default=-1, type=int)
    parser.add_argument('-check', "--check", default=1, type=int)
    args = parser.parse_args()
#    print(args)
    base_path = args.path
    print(base_path)
    max_pixels = args.maxpixel
    print(max_pixels)
    min_pixels = args.minpixel
    print(min_pixels)
    check_only = args.check
    mask_dir = base_path + '/masks'
    images_dir =  base_path + '/images'
#    print(mask_dir)
#    print(images_dir)

    deleted_file_count = 0
    files = glob.glob(os.path.join(mask_dir, '*.png'))
    if len(files) == 0:
        print('No files found in ', mask_dir)
    else:
        print('found %d files in %s' % (len(files), mask_dir))
    for f in files:
        valid_deleted = check_hero_pixel(f, min_pixels, max_pixels)
        if valid_deleted:
#            print(f)
            filename = os.path.splitext(os.path.basename(f))[0]
#            print(filename)
            
            img_filename = get_image_file_name(images_dir, filename)
            if img_filename != '':
                delete_file(mask_dir, filename + '.png', check_only)
#                    print(image_filename)
                delete_file(images_dir, img_filename + '.jpeg', check_only)
                deleted_file_count += 1

    print('Total deleted file number = %d for images and mask each.' % deleted_file_count)

