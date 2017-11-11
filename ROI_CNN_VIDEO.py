# -*- coding: utf-8 -*-
import numpy as np
from cv2 import *
import caffe
import datetime



# this file should be run from {caffe_root}/examples (otherwise change this line)
caffe_root = '/home/hjy/Source/caffe/examples/mnist/'   #根目录
deploy=caffe_root + 'mnist/deploy.prototxt'    #deploy文件
caffe_model=caffe_root + 'mnist/lenet_iter_9380.caffemodel'   #训练好的 caffemodel
labels_filename = caffe_root + 'mnist/test/labels.txt'  #类别名称文件，将数字标签转换回类别名称
labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network

#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

canny_t1=100
canny_t2=200

maxarea=8000
minarea=4000

IMAGE_HEIGHT=600
IMAGE_WIDTH=800

blockSize = 25
constValue = 10

font=FONT_HERSHEY_SIMPLEX
cap=VideoCapture('6__clip.avi')
kernel1=getStructuringElement(MORPH_ELLIPSE,(2,2))
fourcc = VideoWriter_fourcc(*'XVID')
out = VideoWriter('output.avi', fourcc, 20.0, (IMAGE_WIDTH, IMAGE_HEIGHT))
print 'Prework Done'

def predict(roi):
    net.blobs['data'].data[...] = transformer.preprocess('data', roi)
    ### perform classification
    output = net.forward()
    output_prob = net.blobs['Softmax1'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印
    # sort top five predictions from softmax output
    order = output_prob.argsort()[-1]  # 将概率值排序，取出最大值所在的序号
    #print 'the class is:', labels[order]  # 将该序号转换成对应的类别名称，并打印
    return labels[order], np.max(output_prob)

def Process__Proto(image):
    #图像预处理
    img=cvtColor(image,COLOR_BGR2GRAY)
    img_canny=Canny(img,canny_t1,canny_t2)
    img_dilate=dilate(img_canny,kernel1,iterations=1)
    return img_dilate

def ROI_Search(imgThreshold,image):
    numberROI=0
    contours_after_search = []
    img_contours,contours,hierarchy=findContours(imgThreshold,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area=contourArea(cnt)
        if(area < minarea or area >maxarea ):
            continue
        contours_after_search.append(cnt)
    #取出对应ROI并变换成正方形，大小28*28
    for cnt in contours_after_search:
        x, y, w, h = boundingRect(cnt)
        if w/h>3 or w/h<0.3:
            continue
        #x+=10
        #y+=6
        #w-=30
        #h-=14
        rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        numberROI += 1
        ROI= image[y :y + h,x:x + w ]
        #print x, y, w, h
        label, probability=predict(ROI)
        #用于降低背景的影响
        if (probability>0.1):
            putText(image, label, (x, y + 20), font, 0.7, (0, 0, 255), 2)
            #str_pro = '%f' % probability
            #putText(image, str_pro, (x,y+45), font, 0.7, (0, 0, 255), 2)
        rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    str_numROI='Number of Region Proposal:%d' %numberROI
    putText(image, str_numROI, (0,40), font, 1, (255, 0, 0), 2)
    return image

def main():
    numframe=0
    while (1):
        #time = datetime.datetime.now()
        ret, frame = cap.read()
        if (ret != True):
            break
        numframe+=1
        #if (numframe>20):
            #break
        imgThreshold=Process__Proto(frame)
        image=ROI_Search(imgThreshold,frame)
        #time = datetime.datetime.now()-time
        #print time
        imshow('Result', image)
        out.write(image)
        temp = waitKey(15)
        if temp == ord('q'):
            break
        if temp == ord('s'):
            waitKey()
    cap.release()
    out.release()
    destroyAllWindows()


if __name__ == "__main__":
    main()
