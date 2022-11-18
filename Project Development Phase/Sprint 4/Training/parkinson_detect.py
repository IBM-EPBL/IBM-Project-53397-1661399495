#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import cv2
import os
import sys
import pickle


# # Image Pre-processing

# In[2]:


def quantify_image(image):
    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")
    # return the feature vector
    return features


# In[3]:


def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
    # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # quantify the image
        features = quantify_image(image)
        # update the data and labels lists, respectively
        data.append(features)
        labels.append(label)
    # return the data and labels
    return (np.array(data), np.array(labels))


# In[4]:


trainingPath= r"/dataset/spiral/training"
testingPath = r"/dataset/spiral/testing"
print("[INFO] loading data...")


# In[5]:


(X_train, y_train) = load_split(trainingPath)
(X_test, y_test) = load_split(testingPath)


# In[6]:


# encode the labels as integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# # Model Building

# In[7]:


print('[INFO] training the model')
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# In[8]:


testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPaths))
idxs = np.random.choice(idxs, size=(25,), replace=False)
images = []


# In[9]:


# loop over the testing samples
for i in idxs:
    # load the testing image, clone it, and resize it
    image = cv2.imread(testingPaths[i])
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    # pre-process the image in the same manner we did earlier
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


# In[10]:


# quantify the image and make predictions based on the extracted
# features using the last trained Random Forest
features = quantify_image(image)
preds = model.predict([features])
label = le.inverse_transform(preds)[0]
# draw the colored class label on the output image and add it to
# the set of output images
color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
images.append(output)


# In[11]:


# make predictions on the testing data and initialize a dictionary
# to store our computed metrics
predictions = model.predict(X_test)

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(y_test, predictions).flatten()
(tn, fp, fn, tp) = cm
print(cm)
accuracy = (tp + tn) / float(cm.sum())
print(accuracy)


# In[ ]:


# create a montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]

# show the output montage
cv2.imshow("Output", montage)
cv2.waitKey()


# In[ ]:


pickle.dump(model,open('parkinson.pkl','wb'))


# In[ ]:




