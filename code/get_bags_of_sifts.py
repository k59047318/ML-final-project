from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    vocab = pickle.load(open('vocab.pkl','rb')) 
    d = len(vocab) 
    image_feats = [] 
    start_time = time() 
    for path in image_paths: 
        img = np.asarray(Image.open(path), dtype='float32') 
        frames, descriptors = dsift(img, step=[10,10], fast=True) 
        dist = distance.cdist(vocab, descriptors, 'euclidean') 
        feat = np.argmin(dist, axis=0) 
        hist, bins = np.histogram(feat, bins=range(0,d+1)) 
        norm = np.linalg.norm(hist) 
        if norm == 0: 
            image_feats.append(hist) 
        else: 
            image_feats.append(hist/norm)
    image_feats = np.array(image_feats)
    end_time = time() 
    print("It takes %.2f s to extract feature."%(end_time-start_time)) 
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
