import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas import *


'''
Hand-Crafted Gaussian 2D Kernel

param:
    size, sigma, K
return:
    2D Kernel
'''
def createGaussianKernel(size, sigma, K=1):
    
    # Create Distance Grid
    x, y = np.mgrid[-(size//2):(size//2) + 1, -(size//2): (size//2) + 1]
    
    # Calculate Gaussian Distribution
    gaussian_kernel = np.exp(-(x**2+y**2)/(2*sigma**2))
    
    # Normalizing
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    return K * gaussian_kernel


def main():
    # [0] Load image  
    imgCheckerBoard = np.array(Image.open('./checkerboard1024-shaded.tif'))
    imgN1 = np.array(Image.open('./N1.bmp').convert('L'))
    
    # [1] Gaussian Low-Pass Filter
    # [IMG1]
    # Setting Config
    K = 1
    sigma = 64
    kernelSize = 255 
    
    kernelOne = createGaussianKernel(kernelSize, sigma, K)
    #print(DataFrame(kernel)) 
    
    # Convolution
    from scipy import signal
    shaddingPatternOne = signal.convolve2d(imgCheckerBoard, kernelOne, boundary='symm', mode='same')
    
    # Perform img / shaddingPattern
    clearPatternOne = imgCheckerBoard / shaddingPatternOne
    
    # [IMG2]
    # Setting Config
    K = 1
    sigma = 43
    kernelSize = 171

    kernelTwo = createGaussianKernel(kernelSize, sigma, K)
    #print(DataFrame(kernelTwo))
    
    shaddingPatternTwo = signal.convolve2d(imgN1, kernelTwo, boundary='symm', mode='same')
    
    # Perform img / shaddingPattern
    clearPatternTwo = imgN1 / shaddingPatternTwo
    
    # [2] Show Result
    plt.figure(figsize=(10,10))

    plt.subplot(2,3,1)
    plt.imshow(imgCheckerBoard, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(shaddingPatternOne, cmap='gray')
    plt.title('shadding_pattern')
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(clearPatternOne, cmap='gray')
    plt.title('clear_pattern')
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.imshow(imgN1, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(shaddingPatternTwo, cmap='gray')
    plt.title('clear_pattern')
    plt.axis('off')
    
    plt.subplot(2,3,6)
    plt.imshow(clearPatternTwo, cmap='gray')
    plt.title('clear_pattern')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()
