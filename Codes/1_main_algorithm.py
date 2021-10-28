"""
This is the main algorithm which uses OpenCV to create clusters of similar template images

#NOTE:
#'Template image' is the image selected before which is cropped several times to apply TemplateMatching from OpenCV onto 'Other image' 

"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import random

def image_resizing(img1,img2):
 
 """
 This function resizes 2 images to the dimensions of smaller one. If dimensions> 500 after resizing, further resizing is done to reduce computational costs(no improvement in algorithm is observed otherwise) 
 """
 
 if img1.size<img2.size:
  img2= cv2.resize(img2,(img1.shape[1],img1.shape[0]))
 elif img1.size>img2.size:
  img1 = cv2.resize(img1,(img2.shape[1],img2.shape[0]))
 #print(img1.shape[1]/2, img2.shape[0]/2, img1.shape)
 
 if (img1.shape[0]>500) and (img1.shape[1]>500):
  img1= cv2.resize(img1,(round(img1.shape[1]/2),round(img1.shape[0]/2)))
  img2= cv2.resize(img2,(round(img2.shape[1]/2),round(img2.shape[0]/2)))
 
 return img1,img2
 
def choice_making(im_gray,list1,list2,thresh):
 
 """
 this function defines cropping dimensions (l,m) for Template Image and q units are skipped for the next crop.
 """
 
 if (img_gray.shape[0]>thresh) and (img_gray.shape[1]>thresh):
  l = list1[0]
  m = list1[1]
  q =list1[2]
 else:
  l =list2[0]
  m=list2[1]
  q= list2[2]
 
 return l,m,q


def matching_images(img_gray,new_img,list1,list2,thresh,main_thre):
  
  """
  This function preprocesses the Other Image, crops Template Image to get a template, applies Template Matching algorithm and returns the result. Further details are mentioned as comments.
  """
  
  locs_of_img = []

  img2 =cv2.imread('img_fold/%s'%new_img)            
  img2_gray =cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
  print('original sizes are- temp:', img_gray.shape,'another image',img2_gray.shape)
  img_gray1,img2_gray1 = image_resizing(img_gray,img2_gray)
  print('after reisizing temp and img are',img_gray1.shape,img2_gray1.shape)

  l,m,q =choice_making(img_gray1,list1,list2,thresh)     #thresh was selected through trial and error
  print('l,m,q are',l,m,q,'and shape of image is',img_gray1.shape)

  for i in range(0,img_gray1.shape[0]-(l-1),q):
   for j in range(0,img_gray1.shape[1]-(m-1),q):
    template =img_gray1[i:i+l,j:j+m]

    try:
     cv2.imshow('temp',template)
     w,h = template.shape[0],template.shape[1]
     res = cv2.matchTemplate(img2_gray1,template,cv2.TM_CCOEFF_NORMED)
     threshold =0.7
     loc = np.where(res>=threshold)          #If Matching Algorithm gives probability of match>= 0.7, then store in variable loc 
     #print(loc,'is loc)
     for pt in zip(*loc[::-1]):
      cv2.rectangle(img2,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
      #cv2.imwrite('res.png',img2)
      cv2.imshow('result.png',img2)
      cv2.waitKey(1) 

    except cv2.error:
     print('too small')

    if all([yt.size for yt in loc])!=0:   
     #print('its not empty')
     locs_of_img.append(loc)
     
  a= len(range(0,img_gray1.shape[0]-(l-1),q))
  b= len(range(0,img_gray1.shape[1]-(m-1),q))  
  print(a*b, 'is all template tried for this image')
  if len(locs_of_img)>=(main_thre*(a*b)):              #If no. of template matches between 2 pictures> threshold*total no. of templates tried then classify as a match
   print('**************************its a match********************************')  # main_thre is selected through trial and error
  else:
   locs_of_img =[]  
  
  return(locs_of_img) 


train_dir = os.listdir('img_fold')

for ft in range(0,2):           
 
 """
 2 epoches will be run.
  1st epoch : For each image, 250 other images are randomly choosen. If any of 250 images is a match, then the we run through the entire dataset, else we move to the next image.
  2nd epoch : With the images skipped or were not classified with any image, we run the algorithm again for matches. If still there are no matches, manual error  
              analysis will be done and if images are Indian memes then they'll be inputted to a similar model based on pyteserract (OCR). 
  In both epoches if images are detected as similar, both are stored and removed to not repeat their occurence in future iterations.             
 """
 
 print('this is epoch',ft)
 if ft==0:           #ft is epoch no.
  lista =[95,95,70]
  thresh =350
  main_thre=0.67   # both thresholds are trial and error outcomes
 elif ft==1:
  lista=[75,75,60]
  thresh=250
  main_thre=0.66
 listb =[30,30,20]

 print(len(train_dir))
 kh = 1              # kh is image no.
 for im_name in train_dir:
  print('#############################    NEW  TEMPLATE      ######################################')
  print(len(train_dir),'is how long train_dir is')
  print('no is',kh)
  print('epoch no',ft)
  kh+=1
  cluster =[]
  img =cv2.imread('img_fold/%s'%im_name) 
  img_gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
  new_list = random.sample(train_dir,250) # 250 random images are selected
  num =0

  for image in new_list:
   print('here')
   if image!=im_name:
    print(image, im_name)
    locs_of_img = matching_images(img_gray,image,lista,listb,thresh,main_thre)
    print(len(locs_of_img), 'is how long it is')
    if len(locs_of_img)!=0:
     print('*******************','is a match')
     print('detected')
     detected=True
     break  
    else:
     print('check for next image')
     num+=1
     detected =False  
     print(num)
     print('\n')
  print('we\'re here') 
  if not detected:
   continue
  else:
   pass
 
 
  fd = 0
  for new_img in train_dir:         # if any among 250 is detected we move to the original full dataset for match checking
   print('reached the bigger dataset of images')     
   print('imgno',fd)
   fd= fd+1
   locs_of_imgz = matching_images(img_gray,new_img,lista,listb,thresh,main_thre)
   if locs_of_imgz!= [] :
    print(len(locs_of_imgz),'is template matched')
    cluster.append(new_img)         
    #print(new_img,'is being removed')
    if new_img!=im_name:
     print('removing',new_img)
     train_dir.remove(new_img)
     #print('now train_dir is',train_dir)
    else:
     print('same images, hence not removing')

  if len(cluster)<2:        
   print(cluster, 'is the cluster-not identified any other image hence not removing the template image')
  else:
   print('cluster is having more than 2 items hence removing temp image') 
   filem = 'new_clustg'+str(im_name)       # saving the cluster for template image if matches are identified and consequentially removing it
   f = open(filem,'wb')
   pickle.dump(cluster,f)
   f.close()
   print('-------------------------------created--------------------')
   train_dir.remove(im_name)
  print('after this train_dir is',train_dir)
  print('original_img= ',im_name, 'cluster_is', cluster)  
  
print('train_dir at end is',train_dir) 
filw = 'new_clustg'+str(train_dir[0])   #saving the dataset at end of both epoches, error analysis will be the next step
f = open(filw,'wb')
pickle.dump(train_dir,f)
