# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:45:19 2017

@author: Xuandong Xu
"""
from heatmap import heatmap
from train_class import trainer
from window_search import searcher
from scipy import misc
import glob
import numpy as np
import cv2
from scipy.ndimage.measurements import label
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt


train = trainer()
search = searcher()
heatmapper = heatmap()


def pipeline(img):
    #path = './testresult/'
    #filename = 't'
    #filenameheat = 'heat'
    #count = 0
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    #result = search.find_window(img,np.copy(img),train,scale = 1,offset = 200)
    result,bboxs = search.multisearch(img,train,times=5)
    
    heatmapper.recordboxes(bboxs)
    heat = heatmapper.add_heat(heat)
    heat = heatmapper.apply_threshold(heat,12)
    #heat = heat/heat.max()*255
      

    heatmap_img = np.clip(heat, 0, 255)
    #misc.imsave(path+filenameheat+str(count)+'.png',heatmap_img)
    
    labels = label(heatmap_img)
    final_out = heatmapper.draw_labeled_bboxes(np.copy(img), labels)
    #plt.imshow(result,cmap = 'gray')
    tosave = cv2.cvtColor(final_out,cv2.COLOR_BGR2RGB)
    #misc.imsave(path+filename+str(count)+'.jpg',tosave)
    #count += 1
    return tosave
    
if __name__ == "__main__":
    
    video = './project_video'
    white_output = '{}_rendered.mp4'.format(video)
    clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(2,)
    white_clip = clip1.fl_image(pipeline)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    

