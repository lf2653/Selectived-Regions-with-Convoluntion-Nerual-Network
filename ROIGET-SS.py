# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)
from cv2 import *
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import datetime

cap=VideoCapture('6__clip.avi')
def main():
    numframe = 0
    numroi = 0
    totaltime=0
    while(1):
        # loading astronaut image
        ret, img = cap.read()
        if (ret != True):
            break
        numframe += 1
        # perform selective search
        time = datetime.datetime.now()
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)
        time=datetime.datetime.now()-time
        totaltime+=time.seconds
        numroi+=len(regions)
        print(numframe)
        print(time.seconds)
        print(len(regions))
        # draw rectangles on the original image
    cap.release()
    numroi=float(numroi)
    averageroi=numroi/numframe
    print(averageroi)
    print(totaltime)
    print(totaltime/numframe)
if __name__ == "__main__":
    main()
