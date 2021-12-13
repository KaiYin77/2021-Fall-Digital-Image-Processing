### 數位影像處理  DIP Final Project

---

### 電機4C 洪愷尹 0710851

1. ### Framework

   1. Why Choose YOLO v5 instead of YOLOv3 ?
      - v3是2018的paper，相比之下v5則是2020最新的架構
      - v5在輸入端有做幾項training的技巧
        - Mosaic資料增強
          - 對多張圖片進行拼接成為一張新的圖片，傳入到神經網絡中去學習。
        - 自動anchor size計算
          - v5在anchor的部分是使用動態的而非固定式的靜態anchor，於training時依據設定不同的recall去更新anchor的大小。
        - 自適應圖片縮放技巧
          - 在圖片輸入到網路層前的前處理，對於圖片不是簡單的resize，而是進行letterbox的處理。
   2. YOLO v5 (https://github.com/ultralytics/yolov5)
   3. 基於Pytorch Framework實作，其中選用Large的Pre-trained Weight
   4. 官方提供了4種size的model可以供選用，之中是trade-off的關係：

   ![截圖 2021-11-25 下午8.28.14](/Users/Macbook/Library/Application Support/typora-user-images/截圖 2021-11-25 下午8.28.14.jpg)

   ​	其中，YOLOv5s的網路結構為如下：

   ![yolov5模型框架详解_重剑无锋博客-程序员宝宝_yolov5框架- 程序员宝宝](https://img-blog.csdnimg.cn/20210303142212131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE2NzkyMTM5,size_16,color_FFFFFF,t_70)

   ---

   2. ### Difference Between Baseline result and Ours

   - Scene 1: Baseline![截圖 2021-12-13 下午8.20.23](/Users/Macbook/Documents/文件/影像處理/image_processing/project/screenshot/截圖 2021-12-13 下午8.20.23.jpg)

     Scene 1: Ours![截圖 2021-12-13 下午8.20.12](/Users/Macbook/Documents/文件/影像處理/image_processing/project/screenshot/截圖 2021-12-13 下午8.20.12.jpg)

     - 觀察：同一時刻下，可以觀察到，Baseline的方法，在最後一台腳踏車上的車跟人Detection是miss掉的，而我使用的Yolov5架構能有效辨識出來，在People的Recall，以及Bicylce的Recall上是有效提升的。

   - Scene 2: Baseline![截圖 2021-12-13 下午8.28.05](/Users/Macbook/Documents/文件/影像處理/image_processing/project/screenshot/截圖 2021-12-13 下午8.28.05.jpg)

     Scene 2: Ours![截圖 2021-12-13 下午8.27.42](/Users/Macbook/Documents/文件/影像處理/image_processing/project/screenshot/截圖 2021-12-13 下午8.27.42.jpg)

     - 觀察：同一時刻下，可以觀察到，Baseline的方法，在摔車的人上miss detect到兩個人，而在我的方法上，不僅正確的辨識出一個人外，在後方Family廣告招牌的後方部分被Occlude的人都能正確辨識出位置。

---

[HOW TO RUN]

0 : Person

1 : Bike

```bash
python detect.py --source ./video/person_bike.mp4 --weight yolov5l.pt --class 0 1
```

---

[SOURCE CODE]

https://drive.google.com/drive/folders/1AFcVxbjeXLyXCxa4psMJhcpZJZKawW6I?usp=sharing

---

[VIDEO]

https://drive.google.com/file/d/1KJZi-ls3LZg2owY9Qz7K97N6AhZ_15tl/view?usp=sharing

---

