#!/usr/bin/env python
# Note that this needs:
# sudo pip install websocket-client
# not the library called 'websocket'

'''
MIT License

Copyright (c) 2019 Stephen Vu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from utils import *

class laneDetection:
    
    def __init__(self, frame):

        self.frame = frame
        self.check_lane = 0
        
    def processImage(self):

        self.gray_image = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        self.kernel_size = 5
        self.blur_gray = cv2.GaussianBlur(self.gray_image,(self.kernel_size,self.kernel_size),0)
        self.low_threshold = 50
        self.high_threshold = 150
        self.edges = cv2.Canny(self.blur_gray,self.low_threshold,self.high_threshold)
        #cv2.imshow("Edges",self.edges)
        self.mask = np.zeros_like(self.edges)
        self.ignore_mask_color = 255

        self.imshape = self.frame.shape
        #vertices = np.array([[(0,imshape[0] *1 / 2),(imshape[1], imshape[0] * 1 / 2), (imshape[1], imshape[0]), (0,imshape[0])]], dtype=np.int32)
        self.vertices = np.array([[(0,self.imshape[0] * 1 / 2),(self.imshape[1], self.imshape[0] * 1 / 2), (self.imshape[1], self.imshape[0]), (0,self.imshape[0])]], dtype=np.int32)
        cv2.fillPoly(self.mask, self.vertices, self.ignore_mask_color)
        self.masked_edges = cv2.bitwise_and(self.edges, self.mask)
        #cv2.imshow("mask",masked_edges)

        #find all your connected components (white blobs in your image)
        self.nb_components, self.output, self.stats, self.centroids = cv2.connectedComponentsWithStats(self.masked_edges, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        self.sizes = self.stats[1:, -1]; self.nb_components = self.nb_components - 1
        
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 80    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 60 #minimum number of pixels making up a line
        max_line_gap = 30    # maximum gap in pixels between connectable line segments
        self.line_image = np.copy(self.frame)*0 # creating a blank to draw lines on
        
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        self.lines = cv2.HoughLinesP(self.masked_edges, rho, theta, threshold, np.array([]), #modify masked_edges = img2.astype(np.uint8)
                                    min_line_length, max_line_gap)
        
        # Iterate over the output "lines" and draw lines on a blank image

        self.line = None
        try:
            for self.line in self.lines:
                for x1,y1,x2,y2 in self.line:
                    cv2.line(self.line_image,(x1,y1),(x2,y2),(255,0,0),2)
        except:
            pass
            #print("No line to draw")
        # Create a "color" binary image to combine with line image
        #color_edges = np.dstack((edges, edges, edges))wdlog_lineRight
        
        # Draw the lines on the original image
        self.lines_edges = cv2.addWeighted(self.frame, 0.8, self.line_image, 1, 0)
        
        return self.lines_edges,self.lines,self.nb_components

    ######################################## TEST FOR WHITE NOISE ########################################################
    ########################################        BETA          ########################################################
    ########################################       DAT VU         ########################################################
    ######################################################################################################################
    ######################################################################################################################

    def processImage_noise(self):

        self.gray_image = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        self.kernel_size = 5
        self.blur_gray = cv2.GaussianBlur(self.gray_image,(self.kernel_size,self.kernel_size),0)
        self.low_threshold = 50
        self.high_threshold = 150
        self.edges = cv2.Canny(self.blur_gray,self.low_threshold,self.high_threshold)
        self.mask = np.zeros_like(self.edges)
        self.ignore_mask_color = 255

        self.imshape = self.frame.shape
        #vertices = np.array([[(0,imshape[0] *1 / 2),(imshape[1], imshape[0] * 1 / 2), (imshape[1], imshape[0]), (0,imshape[0])]], dtype=np.int32)
        self.vertices = np.array([[(0,0),(self.imshape[1], 0), (230, self.imshape[0] * 1 /2), (80,self.imshape[0] * 1/2)]], dtype=np.int32)
        cv2.fillPoly(self.mask, self.vertices, self.ignore_mask_color)
        self.masked_edges = cv2.bitwise_and(self.edges, self.mask)
        #cv2.imshow("Masked",self.masked_edges)
        # Convert orginal image to bird view 
        self.warp_test = birdView(self.masked_edges,self.transform_matrix_gray['M'])    
        #cv2.imshow("warp",self.warp_test)

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.warp_test, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        
        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        
        #print("Components: ",nb_components)
        #components = nb_components
        
        min_size = 300  
        #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        cv2.imshow("im2",img2)
        #masked_edges = img2
        

        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 80    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 60 #minimum number of pixels making up a line
        max_line_gap = 5    # maximum gap in pixels between connectable line segments
        self.line_image = np.copy(self.frame)*0 # creating a blank to draw lines on
        
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        self.lines = cv2.HoughLinesP(img2.astype(np.uint8), rho, theta, threshold, np.array([]), #modify masked_edges = img2.astype(np.uint8)
                                    min_line_length, max_line_gap)
        
        # Iterate over the output "lines" and draw lines on a blank image

        self.line = None
        try:
            for self.line in self.lines:
                for x1,y1,x2,y2 in self.line:
                    cv2.line(self.line_image,(x1,y1),(x2,y2),(255,0,0),2)
        except:
            print("No line to draw")
        # Create a "color" binary image to combine with line image
        #color_edges = np.dstack((edges, edges, edges))wdlog_lineRight
        
        # Draw the lines on the original image
        self.lines_edges = cv2.addWeighted(self.frame, 0.8, self.line_image, 1, 0)
        cv2.imshow("test",self.lines_edges)

        return self.lines_edges,self.lines,self.nb_components

    def processBirdView(self):

        self.corr_img = self.frame
        self.gray_ex = cv2.cvtColor(self.corr_img,cv2.COLOR_BGR2GRAY)
        #(hMin = 0 , sMin = 0, vMin = 180), (hMax = 178 , sMax = 40, vMax = 255)

        lower = np.array([0,35, 240])
        upper = np.array([255, 255, 255])

        # Create HSV Image and threshold into a range.
        self.hsv = cv2.cvtColor(self.corr_img, cv2.COLOR_BGR2HSV)
        #cv2.imshow("HSV",hsv)
        mask = cv2.inRange(self.hsv, lower, upper)
        self.masked = cv2.bitwise_or(self.gray_ex,self.gray_ex, mask= mask)#(white_mask, yellow_mask)
        #colored_img = cv2.bitwise_and(darkened_img, darkened_img, mask=mask)
        #cv2.imshow("colored",masked)
        min_sz = 200 
        self.cleaned = morphology.remove_small_objects(self.masked.astype('bool'),min_size=min_sz,connectivity=2)
        return self.corr_img,self.gray_ex,self.cleaned,self.masked
    
    def perspective(self):
        src_pts = np.float32([[0,85],[320,85],[320,240],[0,240]])
        dst_pts = np.float32([[0,0],[320,0],[200,240],[120,240]])
        # src_pts = np.float32([[0,0],[320,0],[215,240],[105,240]])
        # dst_pts = np.float32([[0,0],[240,0],[240,320],[0,320]])
        self.transform_matrix = perspective_transform(src_pts,dst_pts) #From utils.py
        self.warped_image = birdView(self.cleaned*1.0,self.transform_matrix['M'])    
        return self.transform_matrix, self.warped_image

    def perspective_noise(self):
        src_pts = np.float32([[0,85],[320,85],[320,240],[0,240]])
        dst_pts = np.float32([[0,0],[320,0],[200,240],[120,240]])
        # src_pts = np.float32([[0,0],[320,0],[215,240],[105,240]])
        # dst_pts = np.float32([[0,0],[240,0],[240,320],[0,320]])
        self.transform_matrix_gray = perspective_transform(src_pts,dst_pts) #From utils.py
        self.warped_gray = birdView(self.gray_ex,self.transform_matrix_gray['M'])    
        
        ######################### REMOVE KHOANG NHIEU #############

        cv2.imshow("Test",self.warped_gray)
        lines_edges,lines,_=self.processImage_noise()
        #print(lines)
        return self.warped_gray    
        

    def processing_traffic_sign(self):

        #self.corr_img = frame

        self.gray_ex = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        #(hMin = 0 , sMin = 0, vMin = 180), (hMax = 178 , sMax = 40, vMax = 255)

        lower = np.array([0,0, 0])
        upper = np.array([179, 150, 30])

        # Create HSV Image and threshold into a range.
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS)
        #cv2.imshow("HSV",hsv)
        mask = cv2.inRange(self.hsv, lower, upper)
        self.masked = cv2.bitwise_or(self.gray_ex,self.gray_ex, mask= mask)#(white_mask, yellow_mask)
        #colored_img = cv2.bitwise_and(darkened_img, darkened_img, mask=mask)
        #cv2.imshow("colored",masked)
        min_sz = 200 
        self.cleaned = morphology.remove_small_objects(self.masked.astype('bool'),min_size=min_sz,connectivity=2)
        src_pts = np.float32([[0,85],[320,85],[320,240],[0,240]])
        dst_pts = np.float32([[0,0],[320,0],[200,240],[120,240]])
        # src_pts = np.float32([[0,0],[320,0],[215,240],[105,240]])
        # dst_pts = np.float32([[0,0],[240,0],[240,320],[0,320]])
        self.transform_matrix = perspective_transform(src_pts,dst_pts) #From utils.py
        self.warped_image = birdView(self.cleaned*1.0,self.transform_matrix['M'])
        self.gray_bird = birdView(self.gray_ex,self.transform_matrix['M'])
        return self.warped_image,self.gray_bird
