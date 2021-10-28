"""
Another version of algorithm based on pytesseract. Created for Indian memes.

"""

import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


listt = ['image3456','image7780',...]   #This list contains the images from OpenCV algorithm which were not detected for match with any other images.  

for igy in listt:
 filename= 'new_clustg'+igy
 pickle_in = open(filename,'rb')
 list_need= pickle.load(pickle_in)
 
 for imt in list_need:
  
  """
  Loading, preprocessing the image
  """
  
  print(imt)
  image_link ='img_fold/'+imt         
  image =cv2.imread(image_link)
  gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  img = cv2.medianBlur(gray,5)
  img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  #kernel = np.ones((5,5),np.uint8) 
  #img =cv2.dilate(img, kernel, iterations = 1)
  #kernel = np.ones((5,5),np.uint8)
  #img=cv2.erode(img, kernel, iterations = 1)
  #opening - erosion followed by dilation 
  #kernel = np.ones((5,5),np.uint8)
  #img =cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  #img=cv2.Canny(img, 100, 200)
 
  img_text = pytesseract.image_to_string(img)
  #print(img_text)
  #plt.imshow(image)
  #plt.show()
  
  """
  running pytesseract on template image to get top 5 large bounding boxes in case no. bounding boxes > 5 and obtaining image with the bounding box 
  """
  
  d = pytesseract.image_to_data(image,output_type=Output.DICT)  
  n_boxes=len(d['level'])
  area_list =[]
  for i in range(n_boxes):
   x,y,w,h = d['left'][i],d['top'][i],d['width'][i],d['height'][i]
   #bbox = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
   if w*h!=0:
    area_list.append([i,w*h,[x,y,w,h]])
  area_list.sort(key =lambda x:x[1], reverse=True)
  #print(area_list,'is beofre')
  for i in area_list:
   #print(i)
   for j in area_list:
    if i!=j:
     if i[2]==j[2]:
      #print(i, j)
      #print('removing', j)
      area_list.remove(j)
      
  print(area_list)
  print('length is',len(area_list))
  if len(area_list)>5:
   top5 = area_list[:5]
   max1,max2,max3,max4,max5=[top5[i][0] for i in range(0,5)]
   bbox1,bbox2,bbox3,bbox4,bbox5 = [top5[i][2] for i in range(0,5)] 
   new_image =image.copy()
   g =0
   for m in [bbox1,bbox2,bbox3,bbox4,bbox5]:
    print('no',m)
    x,y,w,h =m
    print(x,y,w,h)
    cropped = image[y:y+h, x:x+w]        
    cv2.imshow('cropped',cropped)
    cv2.waitKey(34)
    #plt.imshow(cropped)
    #plt.show()
    cv2.imwrite('ocr_resultant_img/'+imt+'_'+str(g)+'.png', cropped)
    g=g+1
    new_image[y:y+h, x:x+w] = 255
    #plt.imshow(new_image)
    #plt.show()
    #cv2.imshow('edited',new_image)
    #cv2.waitKey(360)
   
   for t in area_list[5:] :
    x,y,w,h = t[2]
    cropped = image[y:y+h, x:x+w]
    #plt.imshow(cropped)
    #plt.show()
    new_image[y:y+h, x:x+w] = 255

  else:                             # for case where no. of detected bounding boxes< 5, the same as for above case is done
   for i in range(0,len(area_list)):
     bbox_needed =area_list[i][2]
     new_image =image.copy()
     g =0
     x,y,w,h =bbox_needed
     #print(x,y,w,h)
     cropped = image[y:y+h, x:x+w]
     cv2.imshow('cropped',cropped)
     cv2.waitKey(34)
     #plt.imshow(cropped)
     #plt.show()
     cv2.imwrite('ocr_resultant_img/'+imt+'_'+str(g)+'.png', cropped)
     g=g+1
     new_image[y:y+h, x:x+w] = 255
     #plt.imshow(new_image)
     #plt.show()
     #cv2.imshow('edited',new_image)
     #cv2.waitKey(360)

   
  print('finally')                   
  #plt.imshow(new_image)
  #plt.show()
  cv2.imshow('at last',new_image)
  cv2.waitKey(34)
  cv2.imwrite('ocr_resultant_img/'+imt+'leftover'+'.png', new_image) 

  """
  Images were stored and then run through OpenCV based algorithm for matching. The task here was to detect texts in images- which worked better for Indian Memes.
  """  