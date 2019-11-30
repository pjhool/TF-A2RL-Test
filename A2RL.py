from __future__ import absolute_import
import pickle
import argparse
import numpy as np
import tensorflow as tf

import skimage.io as io

import os

import network
from actions import command2action, generate_bbox, crop_input

from  view_finding.vfn  import *

global_dtype = tf.float32

with open('vfn_rl.pkl', 'rb') as f:
    var_dict = pickle.load(f)

image_placeholder = tf.placeholder(dtype=global_dtype, shape=[None,227,227,3])
global_feature_placeholder = network.vfn_rl(image_placeholder, var_dict)

h_placeholder = tf.placeholder(dtype=global_dtype, shape=[None,1024])
c_placeholder = tf.placeholder(dtype=global_dtype, shape=[None,1024])
action, h, c = network.vfn_rl(image_placeholder, var_dict, global_feature=global_feature_placeholder,
                                                           h=h_placeholder, c=c_placeholder)
sess = tf.Session()

def auto_cropping(origin_image):
    batch_size = len(origin_image)

    terminals = np.zeros(batch_size)
    ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)
    img = crop_input(origin_image, generate_bbox(origin_image, ratios))

    global_feature = sess.run(global_feature_placeholder, feed_dict={image_placeholder: img})
    print('global features ==', global_feature[0][:30])
    h_np = np.zeros([batch_size, 1024])
    c_np = np.zeros([batch_size, 1024])

    while True:
        action_np, h_np, c_np = sess.run((action, h, c), feed_dict={image_placeholder: img,
                                                                    global_feature_placeholder: global_feature,
                                                                    h_placeholder: h_np,
                                                                    c_placeholder: c_np})
        ratios, terminals = command2action(action_np, ratios, terminals)
        bbox = generate_bbox(origin_image, ratios)
        if np.sum(terminals) == batch_size:
            return bbox
        
        img = crop_input(origin_image, bbox)



if __name__ == '__main__':

    tf.reset_default_graph()

    vfn = VFN( mode = 'NO_SPP' )  # NO_SPP, SPP_AVG , SPP_MAX

    parser = argparse.ArgumentParser(description='A2RL: Auto Image Cropping')
    parser.add_argument('--image_path', required=True, help='Path for the image to be cropped')
    parser.add_argument('--save_path', required=True, help='Path for saving cropped image')
    args = parser.parse_args()

    IMG_DIR ='./test_images'
    CROP_IMG_DIR = './test_images_cropped'
    filenames = os.listdir(IMG_DIR)
    for filename in filenames:
        full_filename = os.path.join(IMG_DIR, filename)

        filename_no_ext = os.path.splitext(os.path.basename(full_filename))
        crop_filename = filename_no_ext[0] +'_cropped' + filename_no_ext[1]
        crop_full_filename = os.path.join(CROP_IMG_DIR, crop_filename)
        print(full_filename)
        print ( crop_full_filename )
        im = io.imread(args.image_path).astype(np.float32) / 255
        xmin, ymin, xmax, ymax = auto_cropping([im - 0.5])[0]
        io.imsave(crop_full_filename , im[ymin:ymax, xmin:xmax])

        y = io.imread(full_filename )
        z = io.imread(crop_full_filename  )
        images = [
            # io.imread('test.jpg')[:, :, :3],  # remember to replace with the filename of your test image
            # io.imread('test1.jpg')[:, :, :3],
            # io.imread('source.jpg')[:, :, :3],
            y[:, :, :3] ,
            z[:, :, :3]
        ]
        #from View Finding Network , evaluate score, features for original image file and cropped image file

        scores, features = vfn.evaluate_aesthetics_score(images)

        print(' scores ==', scores)
        print('features ==', features[0][0][:30])






