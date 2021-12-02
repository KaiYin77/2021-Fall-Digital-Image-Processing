import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import math
import cmath
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image", help = "Image to process.", default="lenna-RGB.tif")
args = parser.parse_args()


def showImage(*img):
    title = ['[1] Original',
            '[2] Gradient Image',]
    
    for i in range(len(title)):
        plt.subplot(1,len(title),i+1)
        if img[-1] == "tif":
            plt.imshow(img[i], 'gray')
        else:
            plt.imshow(img[i], 'gray')
        plt.title(title[i])
        plt.axis('off')
    
    plt.show()

def CalculateQuantity(u, v):
    (x, y, z) = u.shape
    gradient = np.zeros((x,y), dtype=np.int32)

    for i in range(x):
        for j in range(y):
            gradient[i][j] = int(u[i][j][0])*int(v[i][j][0]) + \
                             int(u[i][j][1])*int(v[i][j][1]) + \
                             int(u[i][j][2])*int(v[i][j][2])

    return gradient

def CalculateAngle(g_xx, g_yy, g_xy):
    theta = 0.5 * np.arctan2(2*g_xy, (g_xx - g_yy))

    return theta

def F_Transform(g_xx, g_yy, g_xy, theta):

    processed = 0.5 * ((g_xx+g_yy) + (g_xx-g_yy)*np.cos(2*theta) + 2*(g_xy)*np.sin(2*theta)) ** (0.5)
    return processed

def ColorEdgeDetection(img):
    u = cv.Sobel(img, cv.CV_16S, 1, 0)
    v = cv.Sobel(img, cv.CV_16S, 0, 1)
    
    g_xx = CalculateQuantity(u, u)
    g_yy = CalculateQuantity(u, u)
    g_xy = CalculateQuantity(u, v)

    theta = CalculateAngle(g_xx, g_yy, g_xy)

    processed = F_Transform(g_xx, g_yy, g_xy, theta)
    
    return processed

def main():
    # Load img
    path = './' + args.image
    if os.path.isfile(path):
        if path[-3:] == "tif":
            extension = "tif"
            Original = cv.imread(path, cv.IMREAD_UNCHANGED)
            Original = cv.cvtColor(Original, cv.COLOR_BGR2RGB)
        elif path[-3:] == "gif":
            extension = "gif"
            Original = Image.open(path).convert(mode='RGB')
            Original = np.array(Original)
    else :
        print ("The file " + path +" does not exist.")
    
    # Perform Color Edge Detection

    gradient_img = ColorEdgeDetection(Original)

    showImage(Original, gradient_img, extension)
   

if __name__ == '__main__':
    main()
