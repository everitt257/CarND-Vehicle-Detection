# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:18:29 2017

@author: everitt257
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label


class heatmap:
    
    def __init__(self):
        self.previousboxes = []
        self.allboxes = []
    
    def recordboxes(self,bbox_lists):
        # it will record up to 14 list of boxes from past images
        # past that, it will delete the old one and insert new one,
        # just like a queue
        if len(self.previousboxes)<14:
            self.previousboxes.append(bbox_lists)
        else:
            self.previousboxes.remove(self.previousboxes[0])
            self.previousboxes.append(bbox_lists)
        
        # refresh allboxes
        self.allboxes = []
        for boxlist in self.previousboxes:
            self.allboxes += boxlist
        #print('allboxes length',len(self.allboxes))
    
    
    
    def add_heat(self, heatmap):
        # Iterate through list of bboxes
        for box_list in self.allboxes:
            for box in box_list:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
        # Return updated heatmap
        return heatmap
        
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap
    
    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

from train_class import trainer
from scipy import misc
import glob
from window_search import searcher
     
if __name__ == "__main__":
    frames = glob.glob('./mytestpic/*')
    train = trainer()
    search = searcher()
    heatmapper = heatmap()

    path = './testresult/'
    filename = 't'
    filenameheat = 'heat'
    filenameheatpre = 'hpre'
    filenamelabel = 'label'
    count = 0
    for frame in frames:
        print(count)
        img = cv2.imread(frame)
        
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        #result = search.find_window(img,np.copy(img),train,scale = 1,offset = 200)
        result,bboxs = search.multisearch(img,train,times=5)
        
        heatmapper.recordboxes(bboxs)
        heat = heatmapper.add_heat(heat)
        misc.imsave(path+filenameheatpre+str(count)+'.png',heat)
        heat = heatmapper.apply_threshold(heat,14)
        #heat = heat/heat.max()*255
          

        heatmap_img = np.clip(heat, 0, 255)
        misc.imsave(path+filenameheat+str(count)+'.png',heatmap_img)
        
        labels = label(heatmap_img)
        misc.imsave(path+filenamelabel+str(count)+'.png',labels[0])
        final_out = heatmapper.draw_labeled_bboxes(np.copy(img), labels)
        #plt.imshow(result,cmap = 'gray')
        tosave = cv2.cvtColor(final_out,cv2.COLOR_BGR2RGB)
        misc.imsave(path+filename+str(count)+'.jpg',tosave)
        count += 1
        #plt.imshow(cv2.cvtColor(final_out,cv2.COLOR_BGR2RGB))