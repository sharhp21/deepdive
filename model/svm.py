import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

#load the digits dataset
digits = datasets.load_digits()
print('Digits dataset deys \n{}'.format(digits.keys()))

print('dataset target name:\n{}'.format(digits.target_names))
print('shape of dataset: {} \nand target:{}'.format(digits.data.shape, digits.target.shape))
print('shape of the images:{}'.format(digits.images.shape))

#the images are also included in the dataset as digits.images
for i in range(0,4):
    plt.subplot(2, 4, i+1)
    plt.axis('off')
    plt.imshow(digits.images[i])
    plt.title('Training:{}'.format(digits.target[i]))
plt.show()

#SVM
n_samples = len(digits.images)
data_images = digits.images.reshape((n_samples, -1))

X_train, X_test, Y_train, Y_test = train_test_split(data_images, digits.target)
print('Training data and target sizes: \n{}, {}'.format(X_train.shape, Y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(X_test.shape, Y_test.shape))

classifier = svm.SVC(gamma=0.001)
#fit to the training data
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(Y_test, Y_pred)))