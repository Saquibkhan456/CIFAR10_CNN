#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# -----

# # The Data
# 
# CIFAR-10 is a dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.

# In[2]:


from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[3]:


x_train.shape


# In[4]:


x_train[0].shape


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


# FROG
plt.imshow(x_train[0])


# In[7]:


# HORSE
plt.imshow(x_train[12])


# # PreProcessing

# In[8]:


x_train[0]


# In[9]:


x_train[0].shape


# In[10]:


x_train.max()


# In[11]:


x_train = x_train/225


# In[12]:


x_test = x_test/255


# In[13]:


x_train.shape


# In[14]:


x_test.shape


# ## Labels

# In[15]:


from tensorflow.keras.utils import to_categorical


# In[16]:


y_train.shape


# In[17]:


y_train[0]


# In[18]:


y_cat_train = to_categorical(y_train,10)


# In[19]:


y_cat_train.shape


# In[20]:


y_cat_train[0]


# In[21]:


y_cat_test = to_categorical(y_test,10)


# ----------
# # Building the Model

# In[23]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[36]:


model = Sequential()

## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[37]:


model.summary()


# In[38]:


from tensorflow.keras.callbacks import EarlyStopping


# In[39]:


early_stop = EarlyStopping(monitor='val_loss',patience=3)


# In[40]:


model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])


# In[42]:


# Careful, don't overwrite our file!
# model.save('cifar_10epochs.h5')


# In[43]:


losses = pd.DataFrame(model.history.history)


# In[44]:


losses.head()


# In[45]:


losses[['accuracy','val_accuracy']].plot()


# In[46]:


losses[['loss','val_loss']].plot()


# In[27]:


model.metrics_names


# In[47]:


print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))


# In[ ]:





# In[48]:


from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(x_test)


# In[49]:


print(classification_report(y_test,predictions))


# In[53]:


confusion_matrix(y_test,predictions)


# In[54]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
# https://github.com/matplotlib/matplotlib/issues/14751


# # Predicting a given image

# In[65]:


my_image = x_test[16]


# In[66]:


plt.imshow(my_image)


# In[68]:


# SHAPE --> (num_images,width,height,color_channels)
model.predict_classes(my_image.reshape(1,32,32,3))


# In[ ]:


# 5 is DOG
# https://www.cs.toronto.edu/~kriz/cifar.html

