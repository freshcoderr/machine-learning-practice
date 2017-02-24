from marchine_learning import kNN
import numpy

dataset , lables = kNN.createDataset()

testX = numpy.array([1.2,1.1])
k=3
outputlable = kNN.KNNClassify(testX , dataset , lables , k)
print 'Your input is: ', testX , 'and Classify to Class is:' , outputlable

testY = numpy.array([0.2,0.1])
outputlable = kNN.KNNClassify(testY , dataset , lables , k)
print 'Your input is: ', testY , 'and Classify to Class is:' , outputlable