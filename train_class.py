# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:52:36 2017

@author: Xuandong Xu
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import time
from fromlesson import extract_features
from sklearn.utils import shuffle

class trainer:
    def __init__(self):
        self.colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (8,8)
        self.hist_bins = 32
        
        # load sample list for cars and notcars
        # if none found, prepare the list
        try:
            train_list = pickle.load(open( "train_list.p", "rb" ))
            self.newlist = train_list["newlist"]
            self.labels = train_list["labels"]
        except:
            train_list = None
        
        if train_list is None:
            print('Preparing Sample Data')
            self.prepare_data()
        # load pretrained classfier
        # if none found, train the classfier
        try:
            classifier = pickle.load(open( "classifier.p", "rb" ))
            self.svc = classifier["svc"]
            self.X_scaler = classifier["X_scaler"]
        except:
            classifier = None
        
        if classifier is None:
            print('Training classifier')
            self.train_classify()
    
    def r8020(self,*mylists):
        # returns 80% and 20% lists
        return80 = lambda mylist:mylist[:int(len(mylist)*0.8)]
        return20 = lambda mylist:mylist[int(len(mylist)*0.8):]
        return1 = return80(mylists[0])
        return2 = return20(mylists[0])
        for mlist in mylists[1:]:
            return1 += return80(mlist)
            return2 += return20(mlist)
        #int(len(mylists[-1])*0.8)
        return return1,return2
        
    def prepare_data(self):
        returnindex = lambda string,temp:[i for i,item in enumerate(temp) if string in item]
        # extract images from each folder, car samples first
        gti_far = glob.glob('./files/vehicles/GTI_Far/*png')
        gti_left = glob.glob('./files/vehicles/GTI_left/*png')
        gti_middle = glob.glob('./files/vehicles/GTI_MiddleClose/*png')
        gti_right = glob.glob('./files/vehicles/GTI_Right/*png')
        gti_non= glob.glob('./files/non-vehicles/GTI/*png')
        # not-car samples
        kitti = glob.glob('./files/vehicles/KITTI_extracted/*png')
        extra = glob.glob('./files/non-vehicles/Extras/*png')
        
        #split by 80 percent
        gti80,gti20 = self.r8020(gti_far,gti_left,gti_middle,gti_right,gti_non)
        extras80,extras20 = self.r8020(kitti,extra)
        # merge, put 20 percent GTI samples to the last
        newlist = gti80+extras80+gti20+extras20
        nonvehicleindex = returnindex('non-vehicles',newlist)
        labels = np.ones(len(newlist))
        labels[nonvehicleindex] = 0
        self.newlist = newlist
        self.labels = labels
        
#        # old version
#        cars = glob.glob('./files/vehicles/*/*png')
#        notcars = glob.glob('./files/non-vehicles/*/*png')
#        
#
#        self.cars = cars[:8500]
#        self.notcars = notcars[:8500]
        train_lists = {"newlist":self.newlist, "labels":self.labels}
        pickle.dump(train_lists, open("train_list.p", "wb"))
        
    
    def train_classify(self):
        features = extract_features(self.newlist,color_space=self.colorspace, spatial_size=self.spatial_size,
                                        hist_bins=self.hist_bins, orient=self.orient, 
                                        pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, hog_channel=self.hog_channel,
                                        spatial_feat=True, hist_feat=True, hog_feat=True)
        
        #return (car_features,notcar_features)
        
#        # define helper functions to use
#        returnindex = lambda string,temp:[i for i,item in enumerate(temp) if string in item]
#        return80 = lambda mylist:mylist[:int(len(mylist)*0.8)]
#        return20 = lambda mylist:mylist[int(len(mylist)*0.8):]        
#        # 1. process cars first
#        l1 = returnindex('GTI_Far',self.cars)
#        l2 = returnindex('GTI_Left',self.cars)
#        l3 = returnindex('GTI_MiddleClose',self.cars)
#        l4 = returnindex('GTI_Right',self.cars)
#        l5 = returnindex('KITTI_extracted',self.cars)
#        # 2. process not cars secondly
#        l6 = returnindex('GTI',self.notcars)
#        l7 = returnindex('Extras',self.notcars)

        X = np.array(features).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        # Define the labels vector
        y = self.labels
        
#        # Split train and test manually
#        sxl1_t = scaled_X[return80(l1)[0]:return80(l1)[-1]]
#        sxl2_t = scaled_X[return80(l2)[0]:return80(l2)[-1]]
#        sxl3_t = scaled_X[return80(l3)[0]:return80(l3)[-1]]
#        sxl4_t = scaled_X[return80(l4)[0]:return80(l4)[-1]]
#        sxl6_t = scaled_X[return80(l6)[0]:return80(l6)[-1]]
#        sx_gti_cars_train = sxl1_t+sxl2_t+sxl3_t+sxl4_t+sxl6_t
#        
#        
#        # ---
#        sxl1_T = scaled_X[return20(l1)[0]:return20(l1)[-1]]
#        sxl2_T = scaled_X[return20(l2)[0]:return20(l2)[-1]]
#        sxl3_T = scaled_X[return20(l3)[0]:return20(l3)[-1]]
#        sxl4_T = scaled_X[return20(l4)[0]:return20(l4)[-1]]
#        sxl6_T = scaled_X[return20(l6)[0]:return20(l6)[-1]]
#        sx_gti_cars_test = sxl1_T+sxl2_T+sxl3_T+sxl4_T+sxl6_T
#        
#        # ---
#        X_extras = scaled_X[l5[0]:l5[-1]]
#        y_extras = 
        
        
        
        # Split up data into randomized training and test sets
#        rand_state = np.random.randint(0, 100)
#        X_train, X_test, y_train, y_test = train_test_split(
#            scaled_X, y, test_size=0.2, random_state=rand_state)
        totalsize = len(self.newlist)
        X_train, y_train = shuffle(scaled_X[:int(totalsize*0.8)],y[:int(totalsize*0.8)])
        X_test, y_test = shuffle(scaled_X[int(totalsize*0.8):],y[int(totalsize*0.8):])
        
        # Use a linear SVC 
        svc = LinearSVC(C = 0.2)
        #svc = SVC(probability = True)
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        self.svc = svc
        self.X_scaler = X_scaler
        classifier = {'svc':svc, 'X_scaler':X_scaler}
        pickle.dump(classifier,open("classifier.p","wb"))        

    def classify(self,single_features):
        svc = self.svc
        
        return svc.predict(single_features)

#from fromlesson import get_hog_features       
#if __name__ == "__main__":
#    # visualize the hog features
#    
#    cars = glob.glob('./files/vehicles/*/*png')
#    notcars = glob.glob('./files/non-vehicles/*/*png')    
#    icar = np.random.randint(0, len(cars))
#    inotcars = np.random.randint(0, len(notcars))
#    
##    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
##    f.tight_layout()
##    img = cv2.imread(cars[icar])
##    ax1.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
##    
##    img = cv2.imread(notcars[inotcars])
##    ax2.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#
#    temp = trainer()
#    fig = plt.subplots(figsize=(12, 12))
#    plt.subplot(4,4,1)
#    img = cv2.imread(cars[icar])
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
#    plt.imshow(gray,cmap='gray')
#    plt.title('Car Gray')
#    for i in range(3):
#        plt.subplot(4,4,2+i)
#        plt.imshow(img[:,:,i],cmap='gray')
#        plt.title('Car Ch'+str(i+1))
#    plt.subplot(4,4,5)
#    gray_features,gray_hog = get_hog_features(gray,temp.orient,temp.pix_per_cell,temp.cell_per_block,
#                                  True,False)
#        
#    plt.imshow(gray_hog,cmap='gray')
#    plt.title('Car Hog')
#    for i in range(3):
#        plt.subplot(4,4,6+i)
#        ch_features,ch_hog = get_hog_features(img[:,:,i],temp.orient,temp.pix_per_cell,temp.cell_per_block,
#                              True,False)
#        #imgMerge = cv2.merge()
#        plt.imshow(ch_hog,cmap='gray')
#        plt.title('Car Hog Ch'+str(i+1))
#    
#    plt.subplot(4,4,9)
#    img = cv2.imread(notcars[inotcars])
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
#    plt.imshow(gray,cmap='gray')
#    plt.title('Not Car Gray')
#    for i in range(3):
#        plt.subplot(4,4,10+i)
#        plt.imshow(img[:,:,i],cmap='gray')
#        plt.title('Not Car Ch'+str(i+1))
#    plt.subplot(4,4,13)
#    gray_features,gray_hog = get_hog_features(gray,temp.orient,temp.pix_per_cell,temp.cell_per_block,
#                                  True,False)
#        
#    plt.imshow(gray_hog,cmap='gray')
#    plt.title('Not Car Hog')
#    for i in range(3):
#        plt.subplot(4,4,14+i)
#        ch_features,ch_hog = get_hog_features(img[:,:,i],temp.orient,temp.pix_per_cell,temp.cell_per_block,
#                              True,False)
#        #imgMerge = cv2.merge()
#        plt.imshow(ch_hog,cmap='gray')
#        plt.title('Not Car Hog Ch'+str(i+1))
#    plt.tight_layout()
    
    