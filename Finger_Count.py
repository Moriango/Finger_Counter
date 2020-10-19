import cv2
import numpy as np

import pdb

from sklearn.metrics import pairwise

background = None

accumulated_weight = 0.5

top = 20
bottom = 300
right = 300
left = 600

def cumulative_avg(frame, accumulated_weight):
    """
    Given a frame and a previous accumulated weight, computed the weighted average of the image passed in.
    """
    global background
    #creates a new background frame for the first time
    if background is None:
        background = frame.copy().astype('float')
        return None
    #compute wieghted average, accumlate it and update background
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, thresh_min=25):
    global background
    
    # Calculates the Absolute Differentce between the backgroud and the passed in frame
    diff = cv2.absdiff(background.astype('uint8'),frame)
    
    # Apply a threshold to the image so we can grab the foreground
    # We only need the threshold, so we will throw away the first item in the tuple with an underscore
    ret, thresh = cv2.threshold(diff,thresh_min,255,cv2.THRESH_BINARY)
    
    #Grab the external contours form the image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If length of contours list is 0, then we didn't grab any contours!
    if len(contours) == 0:
        return None
    else:
        #Assumes the largest object in the contour is the hand
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresh, hand_segment)
    

def count_fingers(thresholded,hand_segment):
    
    conv_hull =cv2.convexHull(hand_segment)
    
    # TOP
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    distance = pairwise.euclidean_distances([cX,cY],Y=[left,right,top,bottom])[0]
    
    max_distance = distance.max()
    
    radius = int(0.9*max_distance)
    circumfrence = (2*np.pi*radius)

    
    circular_roi = np.zeros(thresholded[:2],dtype='uint8')
    
    cv2.circle(circular_roi,(cX,cY),radius,255,10)
    
    circular_roi = cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)
    
    contours,hierarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    count = 0
    
    for cnt in contours:
        
        (x,y,w,h) = cv2.boundingRect(cnt)
        
        out_of_wrist = (cY + (cY*0.25)) > (y+h)
        
        limit_points = ((circumfrence*0.25) > cnt.shape[0])
        
        if out_of_wrist and limit_points:
            count += 1
            
    return count

cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    
    ret, frame = cam.read()
    if frame is None:
        break
    frame_copy = frame.copy()
    
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    if num_frames < 60:
        calc_accum_avg(gray,accumulated_weight)
        
        if num_frames <= 59:
            cv2.putText(frame_copy,'WAIT. GETTING BACKGROUND',(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
    else:
        
        hand = segment(gray)
        
        if hand is not None:
            
            thresholded , hand_segment = hand
            
            # DRAWS CONTOURS AROUND REAL HAND IN LIVE STREAM
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
            
            fingers = count_fingers(thresholded,hand_segment)
            
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            cv2.imshow('Thresholded',thresholded)
            
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
    
    num_frames += 1
    
    cv2.imshow('Finger Count',frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
