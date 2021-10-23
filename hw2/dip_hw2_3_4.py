from PIL import Image
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image", help = "Image to enhance.", default="Bodybone.bmp")
args = parser.parse_args()

def computeMagnitude(gradientX, gradientY):
    height, width = gradientX.shape
    result = np.empty((height, width))

    # use Approximation M(x,y) ~= abs(x) + abs(y)
    for i in range(height):
        for j in range(width):
            result[i][j] = abs(gradientX[i][j]) + abs(gradientY[i][j])
    
    return result

def Box(img):
    from scipy import signal
    kernel = np.array(
                [[1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1]])
    kernel = kernel / int(25)
    
    result = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    return result

def Sobel(img):
    from scipy import signal 
    kernel_x = np.array(
                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]])
    kernel_y = np.array(
                [[-1, 0, 1],
                 [2, 0, 2],
                 [-1, 0, 1]])
    gradientX = signal.convolve2d(img, kernel_x, boundary='symm', mode='same')
    gradientY = signal.convolve2d(img, kernel_y, boundary='symm', mode='same')

    result = computeMagnitude(gradientX, gradientY)   
    return result

def Laplacian(img, param):
    from scipy import signal
    kernel = np.array(
                [[1, 1, 1],
                 [1, -8, 1],
                 [1, 1, 1]])
    
    result = signal.convolve2d(img, -1*kernel, boundary='symm', mode='same')
    return -1*param*result

def showImage(*img):
    title = ['[1] Original', 
             '[2] after Laplacian', 
             '[3] Lap_img', 
             '[4] after Sobel', 
             '[5] Box_Sob', 
             '[6] Mask', 
             '[7] Sharpen', 
             '[8] gamma']
    
    for i in range(len(title)):
        plt.subplot(1,len(title),i+1)
        plt.imshow(img[i], cmap='gray')
        plt.title(title[i])
        plt.axis('off')

    plt.show()

def HistEq(img):
    hist, bins = np.histogram(img.ravel(), 256, (0,256))

    pdf = hist/img.size
    cdf = pdf.cumsum() 
    
    equ_value = np.around(cdf*255).astype('uint8')
    result = equ_value[img]
    
    return result

def main():
    # Param selection
    c = input("input c for Laplacian: ")  
    gamma = input("input gamma for Power-Law : ")
    
    # Load img 
    img = np.array(Image.open('./' + args.image).convert('L'))
    Lap = Laplacian(img, float(c))
    Sob = Sobel(img)
    
    Lap_img = img + Lap
    Box_Sob = Box(Sob)
    Mask = np.multiply(Box_Sob, Lap_img)
    Sharpen = Lap_img + Mask
    
    gamma_transform = np.array(255* ((Sharpen / 255)**float(gamma)), dtype = 'uint8')
    gamma_transform = Box_Sob + Box(gamma_transform)
    showImage(img, Lap, Lap_img, Sob, Box_Sob, Mask, Sharpen, gamma_transform)

if __name__ == '__main__':
    main()
