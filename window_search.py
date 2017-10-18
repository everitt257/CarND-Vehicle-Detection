# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 00:41:47 2017

@author: Xuandong Xu
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fromlesson import convert_color, get_hog_features, bin_spatial, color_hist

class searcher:
    def __init__(self,times=3):
        self.times= times
    
    def find_window(self,img,draw,param,scale=1.5,ystop=660,offset=0):
#        if isinstance(draw,np.ndarray) == False:
#            raise 
        
        ystart = 400
        #ystop = 660
        #scale = scale
        svc = param.svc
        X_scaler = param.X_scaler
        orient = param.orient
        pix_per_cell = param.pix_per_cell
        cell_per_block = param.cell_per_block
        spatial_size = param.spatial_size
        hist_bins = param.hist_bins
        imgcopy = np.copy(img)
        result,bbox = self.find_cars(imgcopy, draw, ystart, ystop, scale, offset, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        return result,bbox
        
    def find_cars(self, img, draw, ystart, ystop, scale, offset, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        
        draw_img = draw
        bbox = []
        #img = img.astype(np.float32)/255
        x_end = img.shape[1]
        img_tosearch = img[ystart:int(ystop),int(offset):x_end-int(offset),:]
        ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps+1):
            for yb in range(nysteps+1):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
#                print('spatial',spatial_features.shape)
#                print('hist',hist_features.shape)
#                print('hog',hog_features.shape)
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                #test_prediction = svc.predict_proba(test_features)[0][1]
                test_prediction = svc.predict(test_features)
                #print(svc.predict_proba(test_features))
                #print(test_prediction)
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    #y_mean = ((ytop_draw+ystart)+(ytop_draw+ystart+win_draw))/2
                    box = ((int(offset)+xbox_left, ytop_draw+ystart),(int(offset)+xbox_left+win_draw,ytop_draw+win_draw+ystart))
                    bbox.append(box)
                    cv2.rectangle(draw_img,(int(offset)+xbox_left, ytop_draw+ystart),(int(offset)+xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                    
        return draw_img,bbox
        
    def multisearch(self,img,param,times=3):
        # suppose scale is from 1 to 2
        scales = [1.75-i*(1./(times-1)) for i in range(times)]
        ystops = [660-i*(260./times) for i in range(times)]
        xoffsets = [i*(600./times) for i in range(times)]
        #xoffsets = [0,0,0]
        bboxs = []
        draw = np.copy(img)
        for i in range(times):
            draw,bbox = self.find_window(img,draw,param,scales[i],ystops[i],xoffsets[i])
            bboxs.append(bbox)
        
        bboxs = [i for i in bboxs if len(i)>0]
        return draw,bboxs

from train_class import trainer
import glob
from scipy import misc

if __name__ == "__main__":
    frames = glob.glob('./mytestpic/*')
    train = trainer()
    search = searcher()
    path = './testresult/'
    filename = 'search'
    count = 0
    for frame in frames[1:30]:
        print(count)
        img = cv2.imread(frame)        
        result,bboxs = search.multisearch(img,train,times=5)
        saveimg = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        misc.imsave(path+filename+str(count)+'.jpg',saveimg)
        count += 1
#    frame = frames[4]
#    img = cv2.imread(frame)
#    draw = np.copy(img)
#    #result,bboxs = search.find_window(img,draw,train,scale=1.5,ystop=660,offset=120)
#    result,bboxs = search.multisearch(img,train,times=3)
#    plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
#    misc.imsave('./forreportgrid.jpg',cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
