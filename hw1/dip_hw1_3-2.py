import numpy as np
import cv2 as cv
import math
import PIL
from tqdm import tqdm
from matplotlib import pyplot as plt

def calCDF(img):
    hist, bins = np.histogram(img.ravel(), 256, (0,256))

    pdf = hist/img.size
    cdf = pdf.cumsum()
    
    return cdf

def HistEq(img):
    
    cdf = calCDF(img)
    
    equ_value = np.around(cdf*255).astype('uint8')
    result = equ_value[img]
    return result
    
def LocalEnh(img):
    hist, bins = np.histogram(img.ravel(), 256, (0,256))
    pdf = hist/img.size
    
    # Global mean & variance
    print(max(img.ravel()))
    mean = sum((ri * pdf[ri]) for ri in range(256))
    print(mean)
    var = sum((ri - mean) ** 2 * pdf[ri] for ri in range(256))
    std = math.sqrt(var)
    print(std)
    
    # Specified Param
    k0 = 0.045
    k1 = 3
    k2 = 0
    k3 = 0.2
    C = 100

    h, w = img.shape[:2]
    # Local Enchancement
    img_new = img
    width = 3
    m = []
    s = []
    for i in tqdm(range(width//2,h - width//2)):
        for j in range(width//2,w - width//2):
            hist, _ = np.histogram(img[i-width//2:(i+1)+width//2, j-width//2:(j+1)+width//2].ravel(), 256, (0,256))
            pdf = hist / (width**2)

            mxy = sum((ri * pdf[ri]) for ri in range(256))
            sxy = math.sqrt(sum(((ri - mxy)**2) * pdf[ri] for ri in range(256)))
            m.append(mxy)
            s.append(sxy)
            if mxy <= k1*mean and mxy >= k0*mean and sxy <= k3*std and sxy >= k2*std:
                img_new[i,j] = min(img[i,j] * C, 255)
    print(np.mean(m))

    return img_new

##### Start of plt.figure #####
plt.figure(figsize=(5,10))

##### [0] Loading Raw Image #####
plt.subplot(3,2,1)
original = cv.imread('hidden object2.jpg', cv.IMREAD_GRAYSCALE)
plt.imshow(original, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(3,2,2)
plt.hist(original.ravel(), bins=256, range=(0,256), color='k')
plt.title('Original')

##### [1] Histgram Equalization #####
hist_eq = HistEq(original)

plt.subplot(3,2,3)
plt.imshow(hist_eq, cmap='gray')
plt.title('after Histogram Equalization')
plt.axis('off')

plt.subplot(3,2,4)
plt.hist(hist_eq.ravel(), bins=256, range=(0,256), color='k')
plt.title('after Histogram Equalization')

##### [2] Local Enhancement based on Local Histogram Statistics #####
hist_stat = LocalEnh(original)

plt.subplot(3,2,5)
plt.imshow(hist_stat, cmap='gray')
plt.title('Local Enhancement based on Local Histogram Statistics')
plt.axis('off')

plt.subplot(3,2,6)
plt.hist(hist_stat.ravel(), bins=256, range=(0,256), color='k')
plt.title('Local Enhancement based on Local Histogram Statistics')

plt.show()
