from PIL import Image

import pdb
import numpy as np

def get_tiny_images(image_paths):

    '''
    Input : 
        image_paths: a list(N) of string where where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    N = len(image_paths)
    size = 16
    tiny_images = np.zeros((N, size**2)) 
    
    for i in range(N): 
        image = Image.open(image_paths[i]) 
        image = image.resize((size,size), Image.ANTIALIAS) 
        image = np.asarray(image) 
        image = image.flatten() 
        image = (image - np.mean(image))/np.std(image) 
        tiny_images[i, :] = image 
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
