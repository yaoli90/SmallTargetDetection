'''
This is a python implementation of the "small target detection" algorithm in
--------------------------------------------------------------------------------
Chenqiang Gao, Deyu Meng, Yi Yang, et al.,
"Infrared Patch-Image Model for Small Target Detection in a Single Image,"
Image Processing, IEEE Transactions on, vol. 22, no. 12, pp. 4996-5009, 2013.
--------------------------------------------------------------------------------
Please note that this code do NOT contain the segmentation step. If you the
locations of small targets, you have to use a segmentation algorithm.

If you have any questions, please contact:
Author: Yao Li
Email: yaoli0508@hit.edu.cn
Copyright:  Harbin Institute of Technology, College of Mathematics
--------------------------------------------------------------------------------
License: Our code is only available for non-commercial research use.
'''
import numpy as np
from APG_IR import APG_IR

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def mat2gray(image):
    return (image - np.min(image))/np.ptp(image)


'''
Input:
    image: (np.array, dtype=uint8) m x n matrix of an infrared image
    options: (dict)
        'dw': width of the patch
        'dh': height of the patch
        'x_step': sliding steps of patch along x-axis
        'y_step': sliding steps of patch along y-axis
Output:
    A_hat: (np.array, dtype=uint8) m x n matrix estimates for background image
    E_hat: (np.array, dtype=uint8) m x n matrix estimates for target image

% [A_hat, E_hat] - estimates for background image and target image, respectively
'''
def winRPCA_median(image, options):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    m, n = image.shape

    D = []
    for i in range(0, m-options['dh']+1, options['y_step']):
        for j in range(0, n-options['dw']+1, options['x_step']):
            D.append(image[i:i+options['dh'], j:j+options['dw']].flatten('F'))
    D_array = np.array(D).T
    D_normalized = mat2gray(D_array)
    Lambda = 1 / np.sqrt(max(m, n))
    A1, E1 = APG_IR(D_normalized, Lambda)

    AA = np.zeros((m, n, 100))
    EE = np.zeros((m, n, 100))

    index = 0
    C = np.zeros(image.shape)
    A_hat = np.zeros(image.shape)
    E_hat = np.zeros(image.shape)

    for i in range(0, m-options['dh']+1, options['y_step']):
        for j in range(0, n-options['dw']+1, options['x_step']):
            temp = A1[:, index].reshape((options['dw'], options['dh'])).T
            temp1 = E1[:, index].reshape((options['dw'], options['dh'])).T
            C[i:i+options['dh'], j:j+options['dw']] += 1
            index += 1
            for ii in range(i, i+options['dh']):
                for jj in range(j, j+options['dw']):
                    AA[ii, jj, int(C[ii, jj])-1] = temp[ii-i, jj-j]
                    EE[ii, jj, int(C[ii, jj])-1] = temp1[ii-i, jj-j]

    for i in range(m):
        for j in range(n):
            if C[i,j] > 0:
                A_hat[i,j] = np.median(AA[i,j,:int(C[i,j])])
                E_hat[i,j] = np.median(EE[i,j,:int(C[i,j])])

    return A_hat, E_hat
