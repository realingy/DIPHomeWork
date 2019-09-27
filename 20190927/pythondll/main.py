#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/27

from ctypes import *

dll = cdll.LoadLibrary('dlltest.dll')
ret = dll.IntAdd(2, 4)

print(ret)

