#! /usr/bin/env python  
import struct
import numpy as np
# import matplotlib.pyplot as plt
import PIL.Image as Image  #python 3 这样导入  ,python 2 -> import Image 报错，
import sys

print(list(sys.argv))  # sys.argv 只有一个元素 当前目录的文件地址
print(list(sys.path))  # 有多个元素
input_path = '5_2_CNN_MNIST'  #mnist数据库解压后的所在路径
output_path = '5_2_CNN_MNIST'  #生成的图片所在的路径

# =====read labels=====  
label_file = input_path + '/train-labels.idx1-ubyte'
label_fp = open(label_file, 'rb')
label_buf = label_fp.read()

label_index = 0
label_magic, label_numImages = struct.unpack_from('>II', label_buf, label_index)
label_index += struct.calcsize('>II')
labels = struct.unpack_from('>60000B', label_buf, label_index)
"""
有的时候需要用Python处理二进制数据，比如，存取文件，socket操作时. 
这时候，可以使用python的struct模块来完成.可以用 struct来处理C语言中的结构体. 
uppack_from有三个参数，第一个是fmt, 第二个是buf, 第三个是index 
fmt 中 > 可以简单的理解为解析出，不知道packinto时会不会使用 < 
II 是按照两个整形数据来解析

calcsize 表示计算两个整形变量的长度 
 
>60000B 表示按照60000个B Byte 二进制数据来解析
"""
# =====read train images=====  
label_map = {}
train_file = input_path + '/train-images.idx3-ubyte'
train_fp = open(train_file, 'rb')
buf = train_fp.read()

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')
k = 0
for image in range(0, numImages):
    label = labels[k]

    if (label in label_map):
        ids = label_map[label] + 1
        label_map[label] += 1

    else:
        label_map[label] = 0
        ids = 0
    k += 1
    if (label_map[label] > 50):
        continue
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')

    im = np.array(im, dtype='uint8')
    im = im.reshape(28, 28)
    #fig=plt.figure()  
    #plotwindow=fig.add_subplot(111)  
    #plt.imshow(im,cmap='gray')  
    #plt.show()  
    im = Image.fromarray(im)
    print('label, ids = ', label, ids)
    print('output_path = ', output_path)
    im.save(output_path + '/%s_%s.bmp' % (label, ids), 'bmp')