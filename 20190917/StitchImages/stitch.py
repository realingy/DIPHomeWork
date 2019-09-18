from imutils import paths
import numpy as np
import imutils
import cv2

class Stitch():
    def work(self):
        imagePaths = sorted(list(paths.list_images('images')))
        ip = []   # 用于存放按次序的图片，因为拼接对次序敏感
        tmp = imagePaths[0]
        for i in range(len(imagePaths)):
            ip.append(tmp[:12] + str(i) + tmp[13:])
        images = []


        for i, imagePath in enumerate(ip[:]):
            if i%1==0:   # 2为隔一张，不需要隔则设置为1即可
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
            return stitched
        else:
            print("got an error ({})".format(status))


    def affine(self, input):
        # 获取图像大小
        rows, cols = input.shape[:2]

        # 设置图像仿射变换矩阵
        pos1 = np.float32([[50,50], [200,50], [50,200]])
        pos2 = np.float32([[10,100], [200,50], [100,250]])
        M = cv2.getAffineTransform(pos1, pos2)

        # 图像仿射变换
        result = cv2.warpAffine(input, M, (cols, rows))

        # 显示图像
        # cv2.imshow("original", input)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()





