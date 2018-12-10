from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
import operator 

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    k = 5
    dist_matrix = distance.cdist(test_image_feats, train_image_feats, 'euclidean') 
    test_predicts = [] 
    for i in range(test_image_feats.shape[0]): 
        dist = dist_matrix[i, :] 
        dist_sort = np.argsort(dist) 
        neighbors = [] 
        for j in range(k): 
            neighbors.append(dist_sort[j]) 
        classVotes = {} 
        for j in range(k): 
            response = train_labels[neighbors[j]] 
            if response in classVotes: 
                classVotes[response] = classVotes[response] + 1 
            else: 
                classVotes[response] = 1 
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True) 
        test_predicts.append(sortedVotes[0][0]) 
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
