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

from utility import allFilePath
from winRPCA_median import winRPCA_median, mat2gray
import matplotlib.pyplot as plt
import numpy as np
import os




input_dir = './images/'
output_dir = './results/'

options = {'dw': 50, 'dh': 50, 'x_step': 10, 'y_step': 10}

files = allFilePath(input_dir)
files.sort()
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'A'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'E'), exist_ok=True)

for i in range(len(files)):
    print('Processing ', files[i])
    input_image = plt.imread(files[i])
    A, E = winRPCA_median(input_image, options)



    plt.subplot(3, len(files), i+1)
    plt.imshow(input_image)
    plt.gray()
    plt.axis('off')
    plt.title(files[i].split('/')[-1])

    maxv = np.max(input_image.astype(np.float32))
    A_uint8 = (mat2gray(A) * maxv).astype(np.uint8)
    E_uint8 = (mat2gray(E) * 255).astype(np.uint8)
    plt.subplot(3, len(files), i+len(files)+1)
    plt.imshow(A_uint8)
    plt.gray()
    plt.axis('off')

    plt.subplot(3, len(files), i+len(files)*2+1)
    plt.imshow(E_uint8)
    plt.gray()
    plt.axis('off')

    plt.imsave(os.path.join(output_dir, 'A', files[i].split('/')[-1]), A_uint8)
    plt.imsave(os.path.join(output_dir, 'E', files[i].split('/')[-1]), E_uint8)

plt.show()
