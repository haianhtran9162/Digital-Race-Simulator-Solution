#!/usr/bin/env python

import json
from uuid import uuid4
# Note that this needs:
# sudo pip install websocket-client
# not the library called 'websocket'
import websocket
import yaml
from geometry_msgs.msg import  PoseStamped
import rospy
from std_msgs.msg import Header,String,Float32
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from PIL import Image
from skimage import morphology
from collections import deque

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def directional_gradient(img,direction='x',thresh=[0,255]):
    if direction=='x':
        sobel = cv2.Sobel(img,cv2.CV_64F,1,0)
    elif direction=='y':
        sobel = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobel_abs = np.absolute(sobel)#absolute value
    scaled_sobel = np.uint8(sobel_abs*255/np.max(sobel_abs))
    binary_output = np.zeros_like(sobel)
    binary_output[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])] = 1
    return binary_output
def color_binary(img,dst_format='HLS',ch=2,ch_thresh=[0,255]):
    '''
    Color thresholding on channel ch
    img:RGB
    dst_format:destination format(HLS or HSV)
    ch_thresh:pixel intensity threshold on channel ch
    output is binary image
    '''
    if dst_format =='HSV':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        ch_binary = np.zeros_like(img[:,:,int(ch-1)])
        ch_binary[(img[:,:,int(ch-1)]>=ch_thresh[0])&(img[:,:,int(ch-1)]<=ch_thresh[1])] = 1
    else:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        ch_binary = np.zeros_like(img[:,:,int(ch-1)])
        ch_binary[(img[:,:,int(ch-1)]>=ch_thresh[0])&(img[:,:,int(ch-1)]<=ch_thresh[1])] = 1
    return ch_binary
    
def birdView(img,M):
    '''
    Transform image to birdeye view
    img:binary image
    M:transformation matrix
    return a wraped image
    '''
    img_sz = (img.shape[1],img.shape[0])
    img_warped = cv2.warpPerspective(img,M,img_sz,flags = cv2.INTER_LINEAR)
    return img_warped

def perspective_transform(src_pts,dst_pts):
    '''
    perspective transform
    args:source and destiantion points
    return M and Minv
    '''
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)
    return {'M':M,'Minv':Minv}    

def find_centroid(image,peak_thresh,window,showMe):
    '''
    find centroid in a window using histogram of hotpixels
    img:binary image
    window with specs {'x0','y0','width','height'}
    (x0,y0) coordinates of bottom-left corner of window
    return x-position of centroid ,peak intensity and hotpixels_cnt in window
    '''
    #crop image to window dimension 
    mask_window = image[int(window['y0']-window['height']):int(window['y0']),int(window['x0']):int(window['x0']+window['width'])]
    histogram = np.sum(mask_window,axis=0)
    centroid = np.argmax(histogram)
    hotpixels_cnt = np.sum(histogram)
    peak_intensity = histogram[centroid]
    if peak_intensity<=peak_thresh:
        centroid = int(int(window['x0']+window['width']/2))
        peak_intensity = 0
    else:
        centroid = int(int(centroid+window['x0']))
    return (centroid,peak_intensity,hotpixels_cnt)  

def find_starter_centroids(image,x0,peak_thresh,showMe):
    '''
    find starter centroids using histogram
    peak_thresh:if peak intensity is below a threshold use histogram on the full height of the image
    returns x-position of centroid and peak intensity
    '''
    window = {'x0':x0,'y0':image.shape[0],'width':image.shape[1]/2,'height':image.shape[0]/2}  
    # get centroid
    centroid , peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    if peak_intensity<peak_thresh:
        window['height'] = image.shape[0]
        centroid,peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    return {'centroid':centroid,'intensity':peak_intensity}

def run_sliding_window(image,centroid_starter,sliding_window_specs,peak_thresh,showMe):
    
    '''
    Run sliding window from bottom to top of the image and return indexes of the hotpixels associated with lane
    image:binary image
    centroid_starter:centroid starting location sliding window
    sliding_window_specs:['width','n_steps']
        width of sliding window
        number of steps of sliding window alog vertical axis
    return {'x':[],'y':[]}
        coordiantes of all hotpixels detected by sliding window
        coordinates of alll centroids recorded but not used yet!        
    '''
    #Initialize sliding window
    window = {'x0':centroid_starter-int(sliding_window_specs['width']/2),
              'y0':image.shape[0],'width':sliding_window_specs['width'],
              'height':int(image.shape[0]/sliding_window_specs['n_steps'])}
    hotpixels_log = {'x':[],'y':[]}
    centroids_log = []
    for step in range(sliding_window_specs['n_steps']):
        if window['x0']<0: window['x0'] = 0
        if (window['x0']+sliding_window_specs['width'])>image.shape[1]:
            window['x0'] = image.shape[1] - sliding_window_specs['width']
        centroid,peak_intensity,hotpixels_cnt = find_centroid(image,peak_thresh,window,showMe=0)
        if step==0:
            starter_centroid = centroid
        if hotpixels_cnt/(window['width']*window['height'])>0.6:
            window['width'] = window['width']*2
            window['x0']  = int(window['x0']-window['width']/2)
            if (window['x0']<0):window['x0']=0
            if (window['x0']+window['width'])>image.shape[1]:
                window['x0'] = image.shape[1]-window['width']
            centroid,peak_intensity,hotpixels_cnt = find_centroid(image,peak_thresh,window,showMe=0)          
            
        #if showMe:
            #print('peak intensity{}'.format(peak_intensity))
            #print('This is centroid:{}'.format(centroid))
        mask_window = np.zeros_like(image)
        mask_window[window['y0']-window['height']:window['y0'],
                    window['x0']:window['x0']+window['width']]\
            = image[window['y0']-window['height']:window['y0'],
                window['x0']:window['x0']+window['width']]
        
        hotpixels = np.nonzero(mask_window)
        #print(hotpixels_log['x'])
        
        hotpixels_log['x'].extend(hotpixels[0].tolist())
        hotpixels_log['y'].extend(hotpixels[1].tolist())
        # update record of centroid
        centroids_log.append(centroid)
            
        # set next position of window and use standard sliding window width
        window['width'] = sliding_window_specs['width']
        window['x0'] = int(centroid-window['width']/2)
        window['y0'] = window['y0'] - window['height']
    return hotpixels_log
def MahalanobisDist(x,y):
    '''
    Mahalanobis Distance for bi-variate distribution
    
    '''
    covariance_xy = np.cov(x,y,rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i-xy_mean[0] for x_i in x])
    y_diff = np.array([y_i-xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff,y_diff])
    
    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md
    
def MD_removeOutliers(x,y,MD_thresh):
    '''
    remove pixels outliers using Mahalonobis distance
    '''
    MD = MahalanobisDist(x,y)
    threshold = np.mean(MD)*MD_thresh
    nx,ny,outliers = [],[],[]
    for i in range(len(MD)):
        if MD[i]<=threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i)
    return (nx,ny)
def update_tracker(tracker,new_value,allx,ally,bestfit):
    '''
    update tracker(self.bestfit or self.bestfit_real or radO Curv or hotpixels) with new coeffs
    new_coeffs is of the form {'a2':[val2,...],'a1':[va'1,...],'a0':[val0,...]}
    tracker is of the form {'a2':[val2,...]}
    update tracker of radius of curvature
    update allx and ally with hotpixels coordinates from last sliding window
    '''
    if tracker =='bestfit':
        bestfit['a0'].append(new_value['a0'])
        bestfit['a1'].append(new_value['a1'])
        bestfit['a2'].append(new_value['a2'])
    elif tracker == 'bestfit_real':
        bestfit_real['a0'].append(new_value['a0'])
        bestfit_real['a1'].append(new_value['a1'])
        bestfit_real['a2'].append(new_value['a2'])
    elif tracker == 'radOfCurvature':
        radOfCurv_tracker.append(new_value)
    elif tracker == 'hotpixels':
        allx.append(new_value['x'])
        ally.append(new_value['y'])
def polynomial_fit(data):
    '''
    a0+a1 x+a2 x**2
    data:dictionary with x and y values{'x':[],'y':[]}
    '''
    a2,a1,a0 = np.polyfit(data['x'],data['y'],2)
    #print(a2,a1,a0)
    return {'a0':a0,'a1':a1,'a2':a2}
def predict_line(x0,xmax,coeffs):
    '''
    predict road line using polyfit cofficient
    x vaues are in range (x0,xmax)
    polyfit coeffs:{'a2':,'a1':,'a2':}
    returns array of [x,y] predicted points ,x along image vertical / y along image horizontal direction
    '''
    x_pts = np.linspace(x0,xmax-1,num=xmax)
    pred = coeffs['a2']*x_pts**2+coeffs['a1']*x_pts+coeffs['a0']
    return np.column_stack((x_pts,pred))

def compute_radOfCurvature(coeffs,pt):
    return ((1+(2*coeffs['a2']*pt+coeffs['a1'])**2)**1.5)/np.absolute(2*coeffs['a2'])