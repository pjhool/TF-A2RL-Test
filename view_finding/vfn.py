
import numpy as np

import tensorflow as tf

import skimage.io as io

import os

import view_finding.network_vfn  as nw
import skimage.io as io
import skimage.transform as transform

batch_size = 1
global_dtype = tf.float32


class VFN( ):
# In the following example, we are going to feed only one image into the network
    def __init__(self , mode = 'NO_SPP') :


# TODO: Change this if your model file is located somewhere else

        if mode == 'NO_SPP' :  # succc
            snapshot = './view_finding/model-wo-spp'
            SPP = False  ##
            pooling = 'max'
        elif mode == 'SPP_MAX'  :
            snapshot = './view_finding/model-spp-max'
            SPP = True
            pooling = 'max'
        else :   # SPP_AVG
            snapshot = './view_finding/model-spp-avg'
            SPP = True
            pooling = 'avg'


        self.vfn_sess = tf.Session(config=tf.ConfigProto())

#tf.reset_default_graph()
        embedding_dim = 1000
        ranking_loss = 'svm'
        net_data = np.load('./view_finding/alexnet.npy', encoding='bytes').item()
        self.image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size, 227, 227, 3])
        var_dict = nw.get_variable_dict(net_data)


        with tf.variable_scope("ranker") as scope:
            feature_vec = nw.build_alexconvnet(self.image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
            self.score_func = nw.score(feature_vec)
        # load pre-trained model
        saver = tf.train.Saver(tf.global_variables())

        self.vfn_sess.run(tf.global_variables_initializer())
        saver.restore(self.vfn_sess, snapshot)

    def getSession(self):
        return self.vfn_sess


# This is the definition of helper function
    def evaluate_aesthetics_score_orig(self , images):
        scores = np.zeros(shape=(len(images),))
        for i in range(len(images)):
            img = images[i].astype(np.float32)/255
            img_resize = transform.resize(img, (227, 227))-0.5
            img_resize = np.expand_dims(img_resize, axis=0)
            scores[i] , _ = self.vfn_sess.run([self.score_func], feed_dict={ self.image_placeholder: img_resize})[0]
        return scores

    def evaluate_aesthetics_score(self,images):

        scores = np.zeros(shape=(len(images),))
        features = []
        for i in range(len(images)):
            #mg = images[i].astype(np.float32)/255
            #img_resize = transform.resize(img, (227, 227))-0.5
            img = images[i].astype(np.float32)/255 - 0.5
            img_resize = transform.resize(img, (227, 227) , mode='constant')
            img_resize = np.expand_dims(img_resize, axis=0)
            score  , feature = self.vfn_sess.run([ self.score_func ], feed_dict={self.image_placeholder: img_resize})[0]
            scores[i] = score
            features.append( feature)
        return scores , features

    def evaluate_aesthetics_score_resized(self, images):

        scores = np.zeros(shape=(len(images),))
        features = []
        for i in range(len(images)):
            #img = images[i].astype(np.float32)/255
            #img_resize = transform.resize(img, (227, 227))-0.5
            img = images[i].astype(np.float32)
            img_resize = img
            img_resize = np.expand_dims(img_resize, axis=0)
            score  , feature = self.vfn_sess.run([ self.score_func ], feed_dict={self.image_placeholder: img_resize})[0]
            scores[i] = score
            features.append( feature)
        return scores , features



