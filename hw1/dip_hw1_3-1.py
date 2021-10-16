import numpy as np
import cv2 as cv
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

def calculate_lookup(src_cdf, ref_cdf):
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table

def find_correspond_z(array, value):
    minVal = 255
    idx = 0 
    for i in range(256):
      tmp = abs(value-array[i])
      if tmp < minVal:
        minVal = tmp
        idx = i
        if minVal == 0:
          break
    return idx

def HistMatch(img):
    raw_hist, _ = np.histogram(img.ravel(), 256, (0,256))
    raw_pdf = raw_hist/img.size
    raw_cdf = raw_pdf.cumsum()
    equ_value = np.around(raw_cdf*255).astype('uint8')
    
    k=0
    c=1675.540831
    G = np.array([])
    pdf = np.array([])
    for i in range(256):
        j = (i**0.4)/c
        k += j
        pdf = np.append(pdf ,j)
        G = np.append(G, int(k*255))
    ref_cdf = pdf.cumsum()
    
    r_to_z_map = calculate_lookup(raw_cdf, ref_cdf)

    equ_img = equ_value[img]
    result = r_to_z_map[img]
    _hist, _ = np.histogram(result.ravel(), 256, (0,256))
    _pdf = _hist/result.size
    _cdf = calCDF(result)
    
    return result, ref_cdf, _cdf

#### Start of plt.figure() #####
plt.figure(figsize=(8,8))

##### [0] Loading Raw Image #####
plt.subplot(3,2,1)
img = cv.imread('aerial_view.tif', 0)
#print(img.dtype)
#print(len(img.shape))
plt.imshow(img, cmap='gray')
plt.axis('off')

##### [1] Raw Image Histogram #####
# Setting Config
histSize = 256
histRange = (0,256)
accumulate = False

# cv.calHist(images, channels, mask, histSize,  ranges)
hist = cv.calcHist([img], [0], None, [histSize], histRange, accumulate=accumulate)

plt.subplot(3,2,2)
plt.plot(hist, color='k')
plt.title('Original')

###### [2] HistEqual #####
hist_eq = HistEq(img)

plt.subplot(3,2,3)
plt.imshow(hist_eq, cmap='gray')
plt.axis('off')

plt.subplot(3,2,4)
plt.hist(hist_eq.ravel(), bins=256, range=(0,256), color='k')
plt.title('after Histogram Equalization')


###### [3] HistMatch #####
hist_mat, t_cdf, cdf = HistMatch(img)

plt.subplot(3,2,5)
plt.imshow(hist_mat, cmap='gray')
plt.axis('off')

plt.subplot(3,2,6)
plt.hist(hist_mat.ravel(), bins=256, range=(0,256), color='k')
plt.title('after Histogram Matching')

plt.show()

plt.subplot(2,2,1)
plt.plot(t_cdf, color='k')
plt.title('Target CDF')
plt.subplot(2,2,2)
plt.plot(cdf, color='k')
plt.title('CDF of Histogram Matching')
plt.show()

