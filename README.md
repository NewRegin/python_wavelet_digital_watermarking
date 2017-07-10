

[TOC]


## 数字水印


### 概述
[数字水印](https://zh.wikipedia.org/wiki/%E6%95%B8%E4%BD%8D%E6%B5%AE%E6%B0%B4%E5%8D%B0)，是指将特定的信息嵌入数字信号中，数字信号可能是音频、图片或是视频等。若要拷贝有数字水印的信号，所嵌入的信息也会一并被拷贝。数字水印可分为浮现式和隐藏式两种，前者是可被看见的水印（visible watermarking），其所包含的信息可在观看图片或视频时同时被看见。一般来说，浮现式的水印通常包含版权拥有者的名称或标志。右侧的示例图片便包含了浮现式水印。电视台在画面角落所放置的标志，也是浮现式水印的一种。
**隐藏式的水印**是以数字数据的方式加入音频、图片或视频中，但在一般的状况下无法被看见。隐藏式水印的重要应用之一是保护版权，期望能借此避免或阻止数字媒体未经授权的复制和拷贝。隐写术（Steganography）也是数字水印的一种应用，双方可利用隐藏在数字信号中的信息进行沟通。数字照片中的注释数据能记录照片拍摄的时间、使用的光圈和快门，甚至是相机的厂牌等信息，这也是数字水印的应用之一。某些文件格式可以包含这些称为“metadata”的额外信息。


### 性质
**安全性**：水印信息应当难以篡改、难以伪造。
**隐蔽性**：水印对感官不可知觉，水印的嵌入不能影响被保护数据的可用性大大降低。不具备这一特性的水印，称为可见水印（Visible Watermarking）。如电视台播放信号的时候在某个角落经常嵌有它的标志。
**强健性**：水印能够抵御对嵌入后数据的一定操作，而不因为一些细微的操作而磨灭。包括数据的传输中产生的个别位错误，图像或视频、音频的压缩。不具备这一特性的水印，称为脆弱水印（Fragile Watermarking）。
**水印容量**：是指载体可以嵌入水印的信息量。


### 相关技术／工具
#### 成熟水印加密工具
http://steghide.sourceforge.net/index.php
**Install**:
http://www.webm.in/2015/10/install-steghide-centos-6/
**yum 问题处理**：
http://wolfword.blog.51cto.com/4892126/1306203

#### 基于小波变换的数字水印实现（Facebook）
**原理**：https://www.researchgate.net/publication/267988699_Image_Watermarking_Using_3-Level_Discrete_Wavelet_Transform_DWT
**slide**：https://www.slideshare.net/suritd/ppt1-48438386
**Python Script**：
**特性**:
1. 采用 2 次小波变换；
2. 支持水印格式：图像，文字；
3. 支持加水印，解水印；
4. 支持一张图像加多个水印（应对截图／剪切攻击）。
```
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import re
import cv2
import time
import pywt
import argparse
import pygame
import numpy as np
import StringIO
from PIL import Image
pygame.init()

ORIGIN_RATE = 0.999
WATERMARK_RATE = 0.0015
TMP_PATH = "word2pic.png"
# word2img
def opencv_image_from_stringio(watermark_word):
	# 用于设置画布大小和颜色
	img = Image.new("RGB", (512, 512), (255, 255, 255))
	font = pygame.font.Font("msyh.ttf", 50)
	# 用于调整文字颜色和背景颜色
	rtext = font.render(watermark_word, True, (100, 100, 100), (255, 255, 255))
	sio = StringIO.StringIO()
	pygame.image.save(rtext, sio)
	sio.seek(0)
	line = Image.open(sio)
	# 用于调整文字在画布上的位置
	img.paste(line, (200, 200))
 	img.save(TMP_PATH)
    	return cv2.imread(TMP_PATH)

def dwt2_single(img):
	coeffs_1 = pywt.dwt2(img, 'haar', mode='reflect')
	coeffs_2 = pywt.dwt2(coeffs_1[0], 'haar', mode='reflect')
	return coeffs_1, coeffs_2

def dwt2(img1, img2):
	coeffs1_1, coeffs1_2 = dwt2_single(img1)
	coeffs2_1, coeffs2_2 = dwt2_single(img2)
	return coeffs1_1, coeffs1_2, coeffs2_2

def idwt2(img, coeffs1_1_h, coeffs1_2_h):
	cf2 = (img, coeffs1_2_h)
	img = pywt.idwt2(cf2, 'haar', mode='reflect')

	cf1 = (img, coeffs1_1_h)
	img = pywt.idwt2(cf1, 'haar', mode='reflect')
	return img

def channel_embedding(origin_image_chan, watermark_img_chan):
	coeffs1_1, coeffs1_2, coeffs2_2 = dwt2(origin_image_chan, watermark_img_chan)
	embedding_image = cv2.add(cv2.multiply(ORIGIN_RATE, coeffs1_2[0]), cv2.multiply(WATERMARK_RATE, coeffs2_2[0]))
	embedding_image = idwt2(embedding_image, coeffs1_1[1], coeffs1_2[1])
	np.clip(embedding_image, 0, 255, out=embedding_image)
	embedding_image = embedding_image.astype('uint8')
	return embedding_image

def get_watermark(args, flag):
	if flag == "image":
		return cv2.imread(args.watermark)
	else:
		return opencv_image_from_stringio(args.watermark_word)

def img_segment_embedding(watermark_img, origin_image):
	origin_size = origin_image.shape[:2]
	watermark_img = cv2.resize(watermark_img, (origin_size[1], origin_size[0]))
	origin_image_r, origin_image_g, origin_image_b = cv2.split(origin_image)  
	watermark_img_r, watermark_img_g, watermark_img_b = cv2.split(watermark_img)  

	embedding_image_r = channel_embedding(origin_image_r, watermark_img_r)
	embedding_image_g = channel_embedding(origin_image_g, watermark_img_g)
	embedding_image_b = channel_embedding(origin_image_b, watermark_img_b)
 	
	embedding_image = cv2.merge([embedding_image_r, embedding_image_g, embedding_image_b])
	return embedding_image

# 划分若干(num*num)块
def split_img_segments(image, num):
	segments = []
	if num <= 1:
		segments.append(image)
		return segments
	ratio = 1.0/float(num)
	height = image.shape[0]  
    	width = image.shape[1]  
    	pHeight = int(ratio*height)  
    	pHeightInterval = (height-pHeight)/(num-1)  
    	pWidth = int(ratio*width)  
    	pWidthInterval = (width-pWidth)/(num-1)  

    	for i in range(num):  
    	    for j in range(num):  
    	        x = pWidthInterval * i  
    	        y = pHeightInterval * j 
    	        cv2.imwrite('slice.png', image[y:y+pHeight, x:x+pWidth, :])
    	        segments.append(image[y:y+pHeight, x:x+pWidth, :])
    	return segments

# 合并若干块
def merge_img_segments(segments, num, shape):
	if num <= 1:
		return segments[0]
	ratio = 1.0/float(num)
	height =shape[0]  
    	width = shape[1]
    	channel = shape[2]
	image = np.empty([height, width, channel], dtype=int)
	  
    	pHeight = int(ratio*height)  
    	pHeightInterval = (height-pHeight)/(num-1)  
    	pWidth = int(ratio*width)  
    	pWidthInterval = (width-pWidth)/(num-1) 
	cnt = 0
	for i in range(num):  
    	    for j in range(num):  
    	        x = pWidthInterval * i  
    	        y = pHeightInterval * j 
    	        image[y:y+pHeight, x:x+pWidth, :] = segments[cnt]
    	        cnt += 1
	return image

# 加水印 
def embedding(args, flag):
	num = args.image_segments_num
	origin_image = cv2.imread(args.origin)
	watermark_img = get_watermark(args, flag)
	# 划分若干块
	origin_img_segments = split_img_segments(origin_image, num)
	embedding_img_segments = []
	for segment in origin_img_segments:
		embedding_img_segments.append(img_segment_embedding(watermark_img, segment))

	# 合并若干块
	embedding_image = merge_img_segments(embedding_img_segments, num, origin_image.shape)	
	cv2.imwrite(args.embedding, embedding_image)

def channel_extracting(origin_image_chan, embedding_image_chan):
	coeffs1_1, coeffs1_2, coeffs2_2 = dwt2(origin_image_chan, embedding_image_chan)
	extracting_img = cv2.divide(cv2.subtract(coeffs2_2[0], cv2.multiply(ORIGIN_RATE, coeffs1_2[0])), WATERMARK_RATE)
	extracting_img = idwt2(extracting_img, (None, None, None), (None, None, None))
	return extracting_img

def img_segment_extracting(origin_image, embedding_image):
	origin_image_r, origin_image_g, origin_image_b = cv2.split(origin_image)  
	embedding_image_r, embedding_image_g, embedding_image_b = cv2.split(embedding_image)  
	extracting_img_r = channel_extracting(origin_image_r, embedding_image_r)
	extracting_img_g = channel_extracting(origin_image_g, embedding_image_g)
	extracting_img_b = channel_extracting(origin_image_b, embedding_image_b)
 	extracting_img = cv2.merge([extracting_img_r, extracting_img_g, extracting_img_b])
 	return extracting_img

# 解水印
def extracting(args):
	num = args.image_segments_num
	embedding_image = cv2.imread(args.embedding)
	origin_image = cv2.imread(args.origin)
	origin_size = origin_image.shape[:2]
 	embedding_image = cv2.resize(embedding_image, (origin_size[1], origin_size[0]))

	# 划分若干块
	origin_img_segments = split_img_segments(origin_image, num)
	embedding_img_segments = split_img_segments(embedding_image, num)
	extracting_img_segments = []
	for i in range (0, num*num):
		extracting_img_segments.append(img_segment_extracting(origin_img_segments[i], embedding_img_segments[i]))

 	# 合并若干块
 	extracting_img = merge_img_segments(extracting_img_segments, num, origin_image.shape)
	cv2.imwrite(args.extracting, extracting_img)

description = '\n'.join([
        'Compares encode algs using the SSIM metric.',
        '  Example:',
        '   python watermark.py  --opt embedding --origin origin.jpg --watermark watermark.jpg --embedding embedding.jpg'
    ])

parser = argparse.ArgumentParser(
    prog='compare', formatter_class=argparse.RawTextHelpFormatter,
    description=description)
parser.add_argument('--opt', default='embedding', help='embedding or extracting')
parser.add_argument('--origin', default='./samples/test.jpg', help='origin image file, length and width must be a multiple of 8')
parser.add_argument("--watermark", default='./samples/watermark.jpg', help='watermark image file')
parser.add_argument("--watermark_word", default='lzh3', help='watermark words')
parser.add_argument("--embedding", default='./samples/watermarked.jpg', help='embedding image file')
parser.add_argument("--image_segments_num", default=1, type=int, help="The sqrt number of image's segments, may be 1,2,4")
parser.add_argument("--extracting", default='./samples/extract.jpg', help='extracting image file')

args = parser.parse_args()

start = time.time()
if args.opt == 'embedding' :
	embedding(args, "image") 
elif args.opt == 'embedding_word':
	embedding(args, "word")
elif args.opt == 'extracting':
	extracting(args)

print (time.time() - start)

```

**Result**：  
 ![](https://ws1.sinaimg.cn/large/006tNc79gy1fh5jyqrvukj30e80e8nas.jpg)    
**origin_img**  
**水印为图像的场景:**  
```
加水印：
python watermark.py  --opt embedding --origin origin.png --watermark watermark.png --embedding embedding.jpg
解水印：
python watermark.py --opt extracting --origin origin.png --embedding embedding.jpg --extracting extracting.jpg
```

![](https://ws3.sinaimg.cn/large/006tNc79gy1fh5jyqeevkj305k02smx6.jpg)    
**watermark_img**  
   ![](https://ws4.sinaimg.cn/large/006tNc79ly1fh6hy0am43j30e80e8dii.jpg)  
**embedding_img**    
   ![](https://ws3.sinaimg.cn/large/006tNc79ly1fh6hy05fd7j30e80e8t8n.jpg)  
**extracting_img**  
**水印为文字的场景:**：  
 ```
加多个水印(对抗截图／剪切攻击)：  
python watermark.py --opt embedding_word --origin origin.png --watermark_word 'lzh3lzh3' --embedding embedding_word.jpg --image_segments_num 2
解水印：
python watermark.py --opt extracting --origin origin.png --embedding embedding_word.jpg --extracting extracting.jpg --image_segments_num 2
 ```
**watermark_word:** lzh3lzh3 (加2x2个水印)  
![](https://ws2.sinaimg.cn/large/006tNc79ly1fh7ygx31bkj30e80e8wh5.jpg)  
**embedding_img**  
![](https://ws1.sinaimg.cn/large/006tNc79ly1fh7yhowzvpj30e80e8jvx.jpg)  
**extracting_img**  




