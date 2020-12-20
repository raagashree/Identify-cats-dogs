import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

#the images can be downloaded from https://www.microsoft.com/en-us/download/details.aspx?id=54765
SOURCEDIR = "E:/Python projects/PetImages"  # Source images dir path
Categories = ['Cat','Dog']

for category in Categories:
    path = os.path.join(SOURCEDIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap = 'gray')
        plt.show()
        break
    break
print(img_array)
print(img_array.shape)
RESIZE = 50
new_array = cv2.resize(img_array,(RESIZE,RESIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()
