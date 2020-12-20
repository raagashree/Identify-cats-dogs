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


"""
The below code is for identifying Cat & Dog Images through CNN
** start of actual coding for Cats & dogs - CNN **

traning_data = []
def create_training_data():
    SOURCEDIR = "E:/Python projects/PetImages"  #load data from dir
    Categories = ['Cat','Dog']

    for category in Categories:
        try:
            path = os.path.join(SOURCEDIR,category)
            class_num = Categories.index(category)  # have to identify cate from 0/1 not string 
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  #in grey scale
                new_array = cv2.resize(img_array,(RESIZE,RESIZE)) # resize for uniformity & have equal datasets
                traning_data.append([new_array, class_num])
        except Exception as e:
            pass
create_training_data()

print(len(traning_data))
import random
random.shuffle(traning_data)  # shuffle so that it trains with random rather than just dog
for sample in traning_data[:10]:
    print(sample[1])
	
for features, labels in traning_data:
    X.append(features)
    y.append(labels)
X = np.array(X).reshape(-1,RESIZE, RESIZE , 1) #X shd be a array -1 (specifies there can be any no of features), 1 (grey scale) , 3 RGB

import pickle


pickle_out = open("X_pickle", "wb")  # save data
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y_pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X_pickle", "rb")
X = pickle.load(pickle_in)
X[1]

"""
