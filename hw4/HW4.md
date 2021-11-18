### 數位影像處理  DIP Homework Chapter 5_1 Homework

---

### 電機4C 洪愷尹 0710851

- 1 由附圖：

  ![Figure_1](/Users/Macbook/Documents/文件/影像處理/image_processing/HW4/Figure_1.png)

- 2 由附圖：

  ![Figure_2](/Users/Macbook/Documents/文件/影像處理/image_processing/HW4/Figure_2.png)

  

- 3：Please comment and compare your two design filters？

  - 首先可以觀察Inverse filter 和 Weiner Filter，可以發現簡單的Inverse Filter產生的還原圖都是不能用，因為兩者的noise都太大了，都將原圖mask住了，但利用Weiner Filter增加了noise的考量，結果會比較清晰。
  - 1跟2的設計，1的degradation filter是nature的turbulence model，藉由這個假設，去還原霧所帶來的noise，額外cascade一個butterworth的低頻率波去模糊化。2的degradation filter參考防手震的motion model去還原原本的圖片。

---

[SOURCE CODE]

```python
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
import cmath

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image", help = "Image to process.", default="book-cover-blurred.tif")
args = parser.parse_args()


def showImage(*img):
    title = ['[1] Original',
            '[2] Inverse Filter',
             '[3] Weiner Filter']
    
    for i in range(len(title)):
        plt.subplot(1,len(title),i+1)
        if img[-1] == "tif":
            plt.imshow(img[i], cmap='gray')
        else:
            plt.imshow(img[i].astype('uint8'))
        plt.title(title[i])
        plt.axis('off')
    
    plt.show()


def FFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log = np.log(1+mag)
    return fshift, mag, log


def IFFT(img):
    ishift = np.fft.ifftshift(img)
    iimg = np.fft.ifft2(ishift)
    imag = np.abs(iimg)
    return imag


def Dist(u, v, M, N):
    #Calculate distance
    distance = ((u - M / 2)**2 + (v - N / 2)**2)**0.5
    return distance


def CreateTurbulenceFitler(size):
    filter = np.ones(size, dtype=np.cdouble)
    (M,N) = filter.shape
    k = 0.0025
    D0 = 70
    # Cascade a Butterworth Lowpass filter
    for u in range(M):
        for v in range(N):
            filter[u][v] = math.exp(
                                -k *
                                ((u - M/2)**2 + (v - N/2)**2) ** (5/6)
                            ) * (1 / (1 + ((Dist(u, v, M, N) / D0)**40)))
    return filter


def CreateMotionFilter(size):
    filter = np.ones(size, dtype=np.cdouble)
    (M,N) = filter.shape
    a = 0.1
    b = 0.1
    T = 1
    for u in range(M):
        for v in range(N):
            param = math.pi*((u-M/2)*a + (v-N/2)*b)
            if param != 0 :
                filter[u][v] = (T / param) * math.sin(param) * cmath.exp(-1j * param)
            else:
                filter[u][v] = 255
    return filter


def main():
    # Load img
    path = './' + args.image
    if os.path.isfile(path):
        if path[-3:] == "tif":
            extension = "tif"
            Original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif path[-3:] == "jpg":
            extension = "jpg"
            Original = cv2.imread(path)
    else :
        print ("The file " + path +" does not exist.")
    
    
    # FFT
    if extension == "jpg":
        FreqDomain = np.empty(Original.shape, dtype=np.cdouble)
        for i in range(3):
            FreqDomain[:,:,i], Magnitude, LogMagnitude = FFT(Original[:,:,i])
    elif extension == "tif":
        FreqDomain, Magnitude, LogMagnitude = FFT(Original)
    
    # Filter + IFFT
    if args.image == "book-cover-blurred.tif":
        H = CreateMotionFilter(FreqDomain.shape)
        F_inverse = FreqDomain / H
        K = 0.00001
        F_wiener = FreqDomain / H * (H*np.conjugate(H)) / ((H*np.conjugate(H)) + K)
        
        # IFFT
        SpatialDomain_inverse = IFFT(F_inverse)
        SpatialDomain_wiener = IFFT(F_wiener)
    else:
        F_inverse = np.empty(Original.shape, dtype=np.cdouble)
        F_wiener = np.empty(Original.shape, dtype=np.cdouble)
        SpatialDomain_inverse = np.empty(Original.shape)
        SpatialDomain_wiener = np.empty(Original.shape)
        for i in range(3):
            H = CreateTurbulenceFitler(FreqDomain[:,:,i].shape)
            F_inverse[:,:,i] = FreqDomain[:,:,i] / H
            K = 0.005
            F_wiener[:,:,i] = FreqDomain[:,:,i] / H * (H*np.conjugate(H)) / ((H*np.conjugate(H)) + K)

            # IFFT
            SpatialDomain_inverse[:,:,i] = IFFT(F_inverse[:,:,i])
            SpatialDomain_wiener[:,:,i] = IFFT(F_wiener[:,:,i])

    showImage(Original, SpatialDomain_inverse, SpatialDomain_wiener, extension)


if __name__ == '__main__':
    main()
```

---

