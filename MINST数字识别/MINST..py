import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

labels_images = pd.read_csv('train.csv',header=None,sep=',', low_memory=False)
images = np.array(labels_images.iloc[1:, 1:])
labels = np.array(labels_images.iloc[1:, :1])
images = np.array(images, dtype=int)
#print(type(images[0][0]))
for i in range(len(images)):
    for j in range(len(images[i])):
        if images[i][j]>0:
            images[i][j]=1
labels = labels.flatten()
labels = np.array(labels, dtype=int)
x_train, x_test, y_train, y_test = train_test_split(images[:5000], labels[:5000], test_size=0.2)

#print(x_train.shape,y_train.shape)
## knn k=3 socre = 0.927
## svm c=4 gamma=0.01 socre = 0.959
#for i in range(1,10):
clf = svm.SVC(C=4,gamma=0.01)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

test = pd.read_csv('test.csv')
test = np.array(test)
print(len(test))
for i in range(len(test)):
    for j in range(len(test[i])):
        if test[i][j]>0:
            test[i][j]=1
results = clf.predict(test)
print(len(results))
a = [i for i in range(1,len(test)+1)]
df = pd.DataFrame({'ImageId':a,'Label':results})
df.to_csv('result.csv',index=False,sep=',')
