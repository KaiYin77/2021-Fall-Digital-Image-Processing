### 數位影像處理  DIP Homework Chapter 6 Homework

---

### 電機4C 洪愷尹 0710851

- 1 由附圖：

  ![Figure_2](/Users/Macbook/Documents/文件/影像處理/image_processing/HW5_TBD/Figure_2.png)

- 2 由附圖：

  ![Figure_1](/Users/Macbook/Documents/文件/影像處理/image_processing/HW5_TBD/Figure_1.png)

- 3 比較：目標是找到圖篇的邊界！

  方法上兩題的設計是相同的，混合RGB三個通道的Gradient以及最大的變化方向，利用此去設計出Ｆ的變換。

  要 detect 一張圖片的 edge，我們可使用 vector method 以找出該圖片的 梯度(the gradient of the image)。

  1. Follow (6-50)以及(6-51)求出 u, v兩方向的偏微分，實作上，直接使用 Sobel Operator代表微分。
  2. Follow (6-52)、(6-53) 以及(6-54)，得到Gradient的大小值 。
  3. Follow (6- 55) 求出 θ(x, y) 。
  4. Follow (6-56) 利用gxx, gyy, gxy 和 θ(x, y)，求出Fθ(x, y)。
  5. 最後，以Grayscale的方式顯示。

---

[SOURCE CODE]

```python
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

    # Show Images
    showImage(Original, gradient_img, extension)
   

if __name__ == '__main__':
    main()
```

---

