from imutils import paths
import numpy as np
import imutils
import cv2


imagePaths = sorted(list(paths.list_images('images')))
ip = []   # 用于存放按次序的图片，因为拼接对次序敏感
tmp = imagePaths[0]
for i in range(len(imagePaths)):
    ip.append(tmp[:12] + str(i) + tmp[13:])
images = []


for i, imagePath in enumerate(ip[:]):
    if i%2==0:   # 2为隔一张，不需要隔则设置为1即可
        print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.rotate(image, 2)   # 横向旋转，因为拼接对方向敏感
        images.append(image)

import time
a=time.time()
print('stitching images...')
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
print(time.time()-a)

if status == 0:
    cv2.imwrite('res.png', stitched)
else:
    print("got an error ({})".format(status))

