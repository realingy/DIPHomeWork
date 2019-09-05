import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# path = '../data/src/'
path = '../data/reg/'
images = os.listdir(path)

def process_image(img):
    m = np.min(np.min(img))
    M = np.max(np.max(img))
    l = M - m

    r1 = l * 0.5 + m   # 这四个参数你可以根据需要自己调, 只是阈值而已
    r2 = l * 0.7 + m

    s1 = l * 0.495 + m
    s2 = l * 0.705 + m

    a1 = s1 / r1
    a2 = (s2-s1) / (r2-r1)
    a3 = (M-s2) / (M-r2)

    index1 = img < r1
    index3 = img > r2
    index2 = (img > r1) & (img < r2)
    img[index1] = img[index1] * a1
    img[index2] = (img-r1)[index2] * a2 + s1
    img[index3] = (img-r2)[index3] * a3 + s2

    return img

if __name__ == '__main__':
    for i in images:
        img_path = os.path.join(path , i)
        img = Image.open(img_path)
        img = np.array(img, 'f')
        plt.figure()
    #    plt.imshow(img, cmap='gray')
        img = process_image(img)
        plt.figure()
    #    plt.imshow(img, cmap='gray')
        save_path = os.path.join('../data/reg_2', i)
        plt.imsave(save_path, img, cmap='gray')



