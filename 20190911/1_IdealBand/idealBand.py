"""
理想带阻/通滤波模板
"""

import numpy as np

class IdealBand():
    def generate(self, src, w, d0, ftype = 'elimin'):
        r, c = src.shape
        template = np.zeros((r, c), np.float32)  # 构建滤波器
        for i in np.arange(r):
            for j in np.arange(c):
                distance = np.sqrt((i - r / 2) ** 2 + (j - c / 2) ** 2)
                if (d0-w/2) <= distance <= (d0+w/2):
                    template[i, j] = 0
                else:
                    template[i, j] = 1
        if ftype == 'pass': # 带通
            template = 1 - template
        return template



