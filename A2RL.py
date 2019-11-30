from __future__ import absolute_import
import pickle
import argparse
import numpy as np
import pandas as pd
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
    parser.add_argument('--image_path', default='./test_images', help='Path for the image to be cropped')
    parser.add_argument('--save_path', default='./test_images_cropped', help='Path for saving cropped image')
    args = parser.parse_args()

    IMG_DIR = args.image_path     # './test_images'
    CROP_IMG_DIR = args.save_path  #  './test_images_cropped'
    filenames = os.listdir(IMG_DIR)

    global_scores = []
    crop_scores =[]
    img_filenames= []
    crop_img_filenames =[]
    a2rl_results = []
    for img_count, filename in enumerate(filenames) :

        try:
            full_filename = os.path.join(IMG_DIR, filename)
            y = io.imread(full_filename)

            image_x = y.shape[0]
            image_y = y.shape[1]
            # print(' image_filename  read : %s ' % img_filename)

            # checking image aspect ratio
            if ((float(image_x) / image_y > 2) or (float(image_x) / image_y < 0.5)):
                print(' img_count[%d], image_filename : %s ' % ( img_count, filename) )
                print(y.shape)
                continue;
            # print (' y == ' , y.shape , 'y.dim =' ,  y.ndim   )
            if y.ndim != 3:
                print('img_count[%d], img_full_name  = [%s] , [%d]' % ( img_count, filename, y.ndim))
                continue

            # io.imread('test1.jpg')[:, :, :3]  # remember to replace with the filename of your test image
        except Exception as e:
            print(e)
            print(' # Exception Occured ### img_count[%d], img_full_name  = [%s] , [%d]' % ( img_count, filename, y.ndim))
            continue

        full_filename = os.path.join(IMG_DIR, filename)

        filename_no_ext = os.path.splitext(os.path.basename(full_filename))
        crop_filename = filename_no_ext[0] +'_cropped' + filename_no_ext[1]
        crop_full_filename = os.path.join(CROP_IMG_DIR, crop_filename)
        print(full_filename)
        print ( crop_full_filename )
        im = io.imread(full_filename ).astype(np.float32) / 255
        xmin, ymin, xmax, ymax = auto_cropping([im - 0.5])[0]
        io.imsave(crop_full_filename , im[ymin:ymax, xmin:xmax])

        y = io.imread(full_filename )
        z = io.imread(crop_full_filename  )
        images = [
            # io.imread('test.jpg')[:, :, :3],  # remember to replace with the filename of your test image
            # io.imread('source.jpg')[:, :, :3],
            y[:, :, :3] ,
            z[:, :, :3]
        ]
        #from View Finding Network , evaluate score, features for original image file and cropped image file

        scores, features = vfn.evaluate_aesthetics_score(images)

        print(' scores ==', scores)
        #print('features ==', features[0][0][:30])

        global_scores += [scores[0]]

        crop_scores += [scores[1]]
        img_filenames += [full_filename]
        crop_img_filenames +=  [crop_full_filename]

        if scores[1] > scores[0] :
           a2rl_results +=['Good']
        else :
           a2rl_results += ['Bad']


    for index, score in enumerate(global_scores) :
        print( ' A2RL_Result [%s] ,g_score [%f], crop_score [%f] , original filename[%s] , crop_filename[%s] '  % ( a2rl_results[index] , global_scores[index] , crop_scores[index] ,
                                                                                               img_filenames[index] , crop_img_filenames[index]  ) )

    pd_a2rl = pd.DataFrame({ "A2RL_Result": a2rl_results , "global_score":global_scores, "crop_score":crop_scores ,
                            "image_filename":img_filenames , 'crop_img_filename':crop_img_filenames })
    # saving A2RL Results to A2RL_Test.csv
    pd_a2rl.to_csv('A2RL_Test.csv' , encoding='utf-8')


