"""
This algorithm will pick up each cluster detected before and plot the images in the cluster.

"""

import pickle
import os 
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array,array_to_img


train_dir = os.listdir('img_fold')
print(len(train_dir))

def get_img_name(dict_needed):
  
  """
  returns a list with only names of images in a cluster
  """
  
  new_list = []
  for i in dict_needed:
    new_name = i.split('_')[0]
    new_list.append(new_name)
  
  return new_list
    
for im_name in train_dir:
 try:
  filem  = 'new_clustg'+str(im_name)
  pickle_in = open(filem,'rb')
  dict_needed = pickle.load(pickle_in)
  print('**************************new cluster****************************')
  print(len(dict_needed))
  new_list = get_img_name(dict_needed)
  print(im_name,'in', new_list)  
  ar = np.array(new_list)
  plt.figure(figsize=(10,10))          #plotting images in a cluster
  for i in range(0,25*5,25):
   to_plot = ar[i:i+25]
   l =1
   for im_name in to_plot:
    img =load_img(os.path.join('v_meme2_half',im_name))
    img_data = img_to_array(img)
    img_data=img_data/255
    plt.subplot(5,5,l)
    l =l+1
    plt.imshow(img_data)
   plt.show()
     
 except FileNotFoundError:
  #print('not found image') 
  pass
