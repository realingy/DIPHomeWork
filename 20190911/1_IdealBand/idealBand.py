import cv2
import numpy as np
import matplotlib.pyplot as plt

# 带通/带阻滤波器


# 理想的带阻/通滤波模板
def IdealBand(src, w, d0, ftype = 'elimin'):
    template = np.zeros(src.shape, dtype=np.float32)  # 构建滤波器
    r, c = src.shape
    for i in np.arange(r):
        for j in np.arange(c):
            distance = np.sqrt((i - r / 2) ** 2 + (j - c / 2) ** 2)
            if (d0-w/2) <= distance <= (d0+w/2):
                template[i, j] = 0
            else:
                template[i, j] = 1
    if ftype == 'pass': #带通
        template = 1 - template
    return template


src = cv2.imread('test.bmp', 0)

# 快速傅里叶变换算法得到频率分布
f = np.fft.fft2(src)

# 默认结果中心点位置是在左上角,
# 调用fftshift()函数转移到中间位置
fshift = np.fft.fftshift(f)

# fft结果是复数, 其绝对值结果是振幅
fsrc = np.log(np.abs(fshift))

#展示结果
plt.subplot(221), plt.imshow(src, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(222), plt.imshow(fsrc, 'gray'), plt.title('Fourier Image')
plt.axis('off')

fdst = IdealBand(fshift, 10, 0) * fshift

plt.subplot(223), plt.imshow(fdst, 'gray'), plt.title('IdealBand Fourier Image')
plt.axis('off')

#傅里叶逆变换
idst = np.fft.ifftshift(fdst)
dst = np.fft.ifft2(idst)
dst = np.abs(dst)

cv2.imwrite('dst.bmp', dst)

plt.subplot(224), plt.imshow(dst, 'gray'), plt.title('Inverse Fourier Fourier')
plt.axis('off')

plt.show()


