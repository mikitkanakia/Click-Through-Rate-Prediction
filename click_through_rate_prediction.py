# Databricks notebook source
# MAGIC %md # **Click-Through Rate Prediction**
# MAGIC #### The project covers the steps for creating a click-through rate (CTR) prediction pipeline. [Criteo Labs](http://labs.criteo.com/) dataset that was used for a recent [Kaggle competition](https://www.kaggle.com/c/criteo-display-ad-challenge).

# COMMAND ----------

# MAGIC %md ### ** Part 1: Featurize categorical data using one-hot-encoding **

# COMMAND ----------

# MAGIC %md #### ** (1a) One-hot-encoding **
# MAGIC #### We would like to develop code to convert categorical features to numerical ones, and to build intuition, we will work with a sample unlabeled dataset with three data points, with each data point representing an animal. The first feature indicates the type of animal (bear, cat, mouse); the second feature describes the animal's color (black, tabby); and the third (optional) feature describes what the animal eats (mouse, salmon).
# MAGIC #### In a one-hot-encoding (OHE) scheme, we want to represent each tuple of `(featureID, category)` via its own binary feature.  We can do this in Python by creating a dictionary that maps each tuple to a distinct integer, where the integer corresponds to a binary feature. To start, manually enter the entries in the OHE dictionary associated with the sample dataset by mapping the tuples to consecutive integers starting from zero,  ordering the tuples first by featureID and next by category.
# MAGIC #### We'll use OHE dictionaries to transform data points into compact lists of features that can be used in machine learning algorithms.

# COMMAND ----------

# Data for manual OHE
# Note: the first data point does not include any value for the optional third feature
sampleOne = [(0, 'mouse'), (1, 'black')]
sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]
sampleDataRDD = sc.parallelize([sampleOne, sampleTwo, sampleThree])

# COMMAND ----------

sampleOHEDictManual = {}
sampleOHEDictManual[(0,'bear')] = 0
sampleOHEDictManual[(0,'cat')] = 1
sampleOHEDictManual[(0,'mouse')] =2 
sampleOHEDictManual[(1,'black')] = 3
sampleOHEDictManual[(1,'tabby')] = 4
sampleOHEDictManual[(2,'mouse')] = 5
sampleOHEDictManual[(2,'salmon')] = 6

# COMMAND ----------

# MAGIC %md #### ** (1b) Sparse vectors **
# MAGIC #### Data points can typically be represented with a small number of non-zero OHE features relative to the total number of features that occur in the dataset.  By leveraging this sparsity and using sparse vector representations of OHE data, we can reduce storage and computational burdens.  Below are a few sample vectors represented as dense numpy arrays. 

# COMMAND ----------

import numpy as np
from pyspark.mllib.linalg import SparseVector

# COMMAND ----------

aDense = np.array([0., 3., 0., 4.])
aSparse = SparseVector(4,[1,3],[3.,4.])

bDense = np.array([0., 0., 0., 1.])
bSparse = SparseVector(4,[3],[1.])

w = np.array([0.4, 3.1, -1.4, -.5])
print aDense.dot(w)
print aSparse.dot(w)
print bDense.dot(w)
print bSparse.dot(w)

# COMMAND ----------

# MAGIC %md #### **(1c) OHE features as sparse vectors **Any feature that occurs in a point should have the value 1.0.  For example, the `DenseVector` for a point with features 2 and 4 would be `[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]`.

# COMMAND ----------

# Reminder of the sample features
# sampleOne = [(0, 'mouse'), (1, 'black')]
# sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
# sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]

# COMMAND ----------

sampleOneOHEFeatManual = SparseVector(7,[2,3],[1.,1.])
sampleTwoOHEFeatManual = SparseVector(7,[1,4,5],[1.,1.,1.])
sampleThreeOHEFeatManual = SparseVector(7,[0,3,6],[1.,1.,1.])

# COMMAND ----------

# MAGIC %md #### **(1d) Define a OHE function **

# COMMAND ----------

def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
    """Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        You should ensure that the indices used to create a SparseVector are sorted.

    Args:
        rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
        OHEDict (dict): A mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length numOHEFeats with indicies equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.
    """
    
    nrawFeats = len(rawFeats)
    list_index = []
    list_ones = []
    for i in range(nrawFeats):
      list_index.append(OHEDict.get(rawFeats[i]))
      list_ones.append(1.0)
    list_index.sort()
    return SparseVector(numOHEFeats,list_index,list_ones)

# Calculate the number of features in sampleOHEDictManual
numSampleOHEFeats = len(sampleOHEDictManual)

# Run oneHotEnoding on sampleOne
sampleOneOHEFeat = oneHotEncoding(sampleOne, sampleOHEDictManual, numSampleOHEFeats)

print sampleOneOHEFeat

# COMMAND ----------

# MAGIC %md #### **(1e) Apply OHE to a dataset **
# MAGIC #### Finally, use the function to create OHE features for all 3 data points in the sample dataset.

# COMMAND ----------

sampleOHEData = sampleDataRDD.map(lambda x: oneHotEncoding(x, sampleOHEDictManual, numSampleOHEFeats))
print sampleOHEData.collect()

# COMMAND ----------

# MAGIC %md ### ** Part 2: Construct an OHE dictionary **

# COMMAND ----------

# MAGIC %md #### **(2a) Pair RDD of `(featureID, category)` **
# MAGIC #### To start, create an RDD of distinct `(featureID, category)` tuples. In our sample dataset, the 7 items in the resulting RDD are `(0, 'bear')`, `(0, 'cat')`, `(0, 'mouse')`, `(1, 'black')`, `(1, 'tabby')`, `(2, 'mouse')`, `(2, 'salmon')`. Notably `'black'` appears twice in the dataset but only contributes one item to the RDD: `(1, 'black')`, while `'mouse'` also appears twice and contributes two items: `(0, 'mouse')` and `(2, 'mouse')`. 

# COMMAND ----------

sampleDistinctFeats = sampleDataRDD.flatMap(lambda x: x).distinct()

# COMMAND ----------

# MAGIC %md #### ** (2b) OHE Dictionary from distinct features **
# MAGIC #### Next, create an `RDD` of key-value tuples, where each `(featureID, category)` tuple in `sampleDistinctFeats` is a key and the values are distinct integers ranging from 0 to (number of keys - 1).  Then convert this `RDD` into a dictionary, which can be done using the `collectAsMap` action.  Note that there is no unique mapping from keys to values, as all we require is that each `(featureID, category)` key be mapped to a unique integer between 0 and the number of keys.  
# MAGIC #### In our sample dataset, one valid list of key-value tuples is: `[((0, 'bear'), 0), ((2, 'salmon'), 1), ((1, 'tabby'), 2), ((2, 'mouse'), 3), ((0, 'mouse'), 4), ((0, 'cat'), 5), ((1, 'black'), 6)]`.

# COMMAND ----------

sampleOHEDict = (sampleDistinctFeats.zipWithIndex().collectAsMap())
print sampleOHEDict

# COMMAND ----------

# MAGIC %md #### **(2c) Automated creation of an OHE dictionary **

# COMMAND ----------

def createOneHotDict(inputData):
    """Creates a one-hot-encoder dictionary based on the input data.

    Args:
        inputData (RDD of lists of (int, str)): An RDD of observations where each observation is
            made up of a list of (featureID, value) tuples.

    Returns:
        dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
            unique integers.
    """
    
    return inputData.flatMap(lambda x: x).distinct().zipWithIndex().collectAsMap()

sampleOHEDictAuto = createOneHotDict(sampleDataRDD)
print sampleOHEDictAuto

# COMMAND ----------

# MAGIC %md ### **Part 3: Parse CTR data and generate OHE features**

# COMMAND ----------

# MAGIC %md #### Before we can proceed, you'll first need to obtain the Criteo data from Criteo. For your convenience, it is hosted [here](https://www.dropbox.com/s/rf64jk6eufmserm/dac_sample.txt?dl=1) in my DropBox Account. Open in a separate browser tab.

# COMMAND ----------

import os
import sys
import os.path
import pyspark
import urllib2

response = urllib2.urlopen('https://www.dropbox.com/s/rf64jk6eufmserm/dac_sample.txt?dl=1')

dacContents = response.read().split('\n')
dacContents = [x.strip().replace('\t', ',') for x in dacContents]

numPartitions = 2
rawData = sc.parallelize(dacContents, numPartitions)

# COMMAND ----------

# MAGIC %md #### **(3a) Loading and splitting the data **
# MAGIC #### We are now ready to start working with the actual CTR data, and our first task involves splitting it into training, validation, and test sets.

# COMMAND ----------

weights = [.8, .1, .1]
seed = 42
rawTrainData, rawValidationData, rawTestData = rawData.randomSplit(weights,seed)
# Cache the data
rawTrainData.cache()
rawValidationData.cache()
rawTestData.cache()

nTrain = rawTrainData.count()
nVal = rawValidationData.count()
nTest = rawTestData.count()
print nTrain, nVal, nTest, nTrain + nVal + nTest
print rawData.take(1)

# COMMAND ----------

# MAGIC %md #### ** (3b) Extract features **
# MAGIC #### We will now parse the raw training data to create an RDD that we can subsequently use to create an OHE dictionary.

# COMMAND ----------

def parsePoint(point):
    """Converts a comma separated string into a list of (featureID, value) tuples.

    Note:
        featureIDs should start at 0 and increase to the number of features - 1.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.

    Returns:
        list: A list of (featureID, value) tuples.
    """
    
    list_features_all = point.split(",")
    list_final = []
    n = len(list_features_all)
    for i in range(1,n):
      list_final.append((i-1,list_features_all[i]))
    
    return list_final

parsedTrainFeat = rawTrainData.map(parsePoint)

numCategories = (parsedTrainFeat
                 .flatMap(lambda x: x)
                 .distinct()
                 .map(lambda x: (x[0], 1))
                 .reduceByKey(lambda x, y: x + y)
                 .sortByKey()
                 .collect())

print numCategories[2][1]


# COMMAND ----------

# MAGIC %md #### **(3c) Create an OHE dictionary from the dataset **
# MAGIC ####Note that we will assume for simplicity that all features in our CTR dataset are categorical.

# COMMAND ----------

ctrOHEDict = createOneHotDict(parsedTrainFeat)
numCtrOHEFeats = len(ctrOHEDict.keys())
print numCtrOHEFeats
print ctrOHEDict[(0, '')]

# COMMAND ----------

# MAGIC %md #### ** (3d) Apply OHE to the dataset **

# COMMAND ----------

from pyspark.mllib.regression import LabeledPoint

# COMMAND ----------

def parseOHEPoint(point, OHEDict, numOHEFeats):
    """Obtain the label and feature vector for this raw observation.

    Note:
        You must use the function `oneHotEncoding` in this implementation or later portions
        of this lab may not function as expected.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.
        OHEDict (dict of (int, str) to int): Mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The number of unique features in the training dataset.

    Returns:
        LabeledPoint: Contains the label for the observation and the one-hot-encoding of the
            raw features based on the provided OHE dictionary.
    """
    
    list_features_all = point.split(",")
    list_final = []
    n = len(list_features_all)
    for i in range(1,n):
      list_final.append((i-1,list_features_all[i]))
    
    if (list_features_all[0] == "0"):
      lbp = LabeledPoint(0,oneHotEncoding(list_final, ctrOHEDict, numCtrOHEFeats))
    else:
      lbp = LabeledPoint(1,oneHotEncoding(list_final, ctrOHEDict, numCtrOHEFeats))
    
    return lbp

OHETrainData = rawTrainData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats))
OHETrainData.cache()
print OHETrainData.take(1)

# Check that oneHotEncoding function was used in parseOHEPoint
backupOneHot = oneHotEncoding
oneHotEncoding = None
withOneHot = False
try: parseOHEPoint(rawTrainData.take(1)[0], ctrOHEDict, numCtrOHEFeats)
except TypeError: withOneHot = True
oneHotEncoding = backupOneHot


# COMMAND ----------

# MAGIC %md #### **Visualization 1: Feature frequency **
# MAGIC #### We will now visualize the number of times each of the 233,286 OHE features appears in the training data. We first compute the number of times each feature appears, then bucket the features by these counts.  The buckets are sized by powers of 2, so the first bucket corresponds to features that appear exactly once (2^0), the second to features that appear twice (2^1), the third to features that occur between three and four (2^2) times, the fifth bucket is five to eight (2^3) times and so on. The scatter plot below shows the logarithm of the bucket thresholds versus the logarithm of the number of features that have counts that fall in the buckets.

# COMMAND ----------

def bucketFeatByCount(featCount):
    """Bucket the counts by powers of two."""
    for i in range(11):
        size = 2 ** i
        if featCount <= size:
            return size
    return -1

featCounts = (OHETrainData
              .flatMap(lambda lp: lp.features.indices)
              .map(lambda x: (x, 1))
              .reduceByKey(lambda x, y: x + y))
featCountsBuckets = (featCounts
                     .map(lambda x: (bucketFeatByCount(x[1]), 1))
                     .filter(lambda (k, v): k != -1)
                     .reduceByKey(lambda x, y: x + y)
                     .collect())
print featCountsBuckets

# COMMAND ----------

import matplotlib.pyplot as plt

x, y = zip(*featCountsBuckets)
x, y = np.log(x), np.log(y)

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(4, 14, 2))
ax.set_xlabel(r'$\log_e(bucketSize)$'), ax.set_ylabel(r'$\log_e(countInBucket)$')
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
pass

display(fig)

# COMMAND ----------

# MAGIC %md #### **(3e) Handling unseen features **
# MAGIC #### However, we must be careful, as some categorical values will likely appear in new data that did not exist in the training data. To deal with this situation, update the `oneHotEncoding()` function to ignore previously unseen categories, and then compute OHE features for the validation data.

# COMMAND ----------

def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
    """Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        If a (featureID, value) tuple doesn't have a corresponding key in OHEDict it should be
        ignored.

    Args:
        rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
        OHEDict (dict): A mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length numOHEFeats with indicies equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.
    """
    nrawFeats = len(rawFeats)
    list_index = []
    list_ones = []
    for i in range(nrawFeats):
      if OHEDict.has_key(rawFeats[i]):
        list_index.append(OHEDict.get(rawFeats[i]))
        list_ones.append(1.0)
    list_index.sort()
    return SparseVector(numOHEFeats,list_index,list_ones)

OHEValidationData = rawValidationData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats))
OHEValidationData.cache()
print OHEValidationData.take(1)

# COMMAND ----------

# MAGIC %md ### ** Part 4: CTR prediction and logloss evaluation **

# COMMAND ----------

# MAGIC %md #### ** (4a) Logistic regression **
# MAGIC #### We are now ready to train our first CTR classifier.  A natural classifier to use in this setting is logistic regression, since it models the probability of a click-through event rather than returning a binary response, and when working with rare events, probabilistic predictions are useful.

# COMMAND ----------

from pyspark.mllib.classification import LogisticRegressionWithSGD

# fixed hyperparameters
numIters = 50
stepSize = 10.
regParam = 1e-6
regType = 'l2'
includeIntercept = True

# COMMAND ----------

model0 = LogisticRegressionWithSGD.train(data = OHETrainData,iterations = numIters, step = stepSize, regParam= regParam,regType = regType, intercept =includeIntercept )
sortedWeights = sorted(model0.weights)
print sortedWeights[:5], model0.intercept

# COMMAND ----------

# MAGIC %md #### ** (4b) Log loss **
# MAGIC #### We will use log loss to evaluate the quality of models.

# COMMAND ----------

from math import log

def computeLogLoss(p, y):
    """Calculates the value of log loss for a given probabilty and label.

    Note:
        log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
        and when p is 1 we need to subtract a small value (epsilon) from it.

    Args:
        p (float): A probabilty between 0 and 1.
        y (int): A label.  Takes on the values 0 and 1.

    Returns:
        float: The log loss value.
    """
    epsilon = 10e-12
    
    if p==0:
      p+=epsilon
    elif p==1:
      p-=epsilon
    
    if y==1:
      logloss = -log(p)
    elif y==0:
      logloss = -log(1-p)
    
    return logloss
  

print computeLogLoss(.5, 1)
print computeLogLoss(.5, 0)
print computeLogLoss(.99, 1)
print computeLogLoss(.99, 0)
print computeLogLoss(.01, 1)
print computeLogLoss(.01, 0)
print computeLogLoss(0, 1)
print computeLogLoss(1, 1)
print computeLogLoss(1, 0)

# COMMAND ----------

# MAGIC %md #### ** (4c)  Baseline log loss **
# MAGIC #### A very simple yet natural baseline model is one where we always make the same prediction independent of the given datapoint, setting the predicted value equal to the fraction of training points that correspond to click-through events (i.e., where the label is one). The value (which is simply the mean of the training labels) is computer and used to compute the training log loss for the baseline model.  The log loss for multiple observations is the mean of the individual log loss values.

# COMMAND ----------

# Note that our dataset has a very high click-through rate by design
# In practice click-through rate can be one to two orders of magnitude lower
classOneFracTrain = OHETrainData.map(lambda x: x.label).mean()
print classOneFracTrain

logLossTrBase = OHETrainData.map(lambda x: computeLogLoss(classOneFracTrain,x.label) ).mean()
print 'Baseline Train Logloss = {0:.3f}\n'.format(logLossTrBase)

# COMMAND ----------

# MAGIC %md #### ** (4d) Predicted probability **
# MAGIC #### The function that computes the raw linear prediction from this logistic regression model and then passes it through a [sigmoid function](http://en.wikipedia.org/wiki/Sigmoid_function) $$ \scriptsize \sigma(t) = (1+ e^{-t})^{-1} $$ to return the model's probabilistic prediction. Then compute probabilistic predictions on the training data.
# MAGIC #### Note that when incorporating an intercept into our predictions, we simply add the intercept to the value of the prediction obtained from the weights and features.  Alternatively, if the intercept was included as the first weight, we would need to add a corresponding feature to our data where the feature has the value one.  This is not the case here.

# COMMAND ----------

from math import exp #  exp(-t) = e^-t

def getP(x, w, intercept):
    """Calculate the probability for an observation given a set of weights and intercept.

    Note:
        We'll bound our raw prediction between 20 and -20 for numerical purposes.

    Args:
        x (SparseVector): A vector with values of 1.0 for features that exist in this
            observation and 0.0 otherwise.
        w (DenseVector): A vector of weights (betas) for the model.
        intercept (float): The model's intercept.

    Returns:
        float: A probability between 0 and 1.
    """
    
    rawPrediction = x.dot(w) + intercept
    
    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    return 1/(1+exp(-rawPrediction))

trainingPredictions = OHETrainData.map(lambda x: getP(x.features, model0.weights, model0.intercept))

print trainingPredictions.take(5)

# COMMAND ----------

# MAGIC %md #### ** (4e) Evaluate the model **

# COMMAND ----------

def evaluateResults(model, data):
    """Calculates the log loss for the data given the model.

    Args:
        model (LogisticRegressionModel): A trained logistic regression model.
        data (RDD of LabeledPoint): Labels and features for each observation.

    Returns:
        float: Log loss for the data.
    """
    
    return data.map(lambda x: computeLogLoss(getP(x.features, model.weights, model.intercept),x.label)).mean()
    
logLossTrLR0 = evaluateResults(model0, OHETrainData)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossTrBase, logLossTrLR0))

# COMMAND ----------

# MAGIC %md #### ** (4f) Validation log loss **
# MAGIC #### Notably, the baseline model for the validation data should still be based on the label fraction from the training dataset.

# COMMAND ----------

logLossValBase = OHEValidationData.map(lambda x: computeLogLoss(classOneFracTrain,x.label) ).mean()

logLossValLR0 = evaluateResults(model0, OHEValidationData)
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, logLossValLR0))

# COMMAND ----------

# MAGIC %md #### **Visualization 2: ROC curve **
# MAGIC #### We will now visualize how well the model predicts our target.  To do this we generate a plot of the ROC curve.  The ROC curve shows us the trade-off between the false positive rate and true positive rate, as we liberalize the threshold required to predict a positive outcome.  A random model is represented by the dashed line.

# COMMAND ----------

labelsAndScores = OHEValidationData.map(lambda lp:
                                            (lp.label, getP(lp.features, model0.weights, model0.intercept)))
labelsAndWeights = labelsAndScores.collect()
labelsAndWeights.sort(key=lambda (k, v): v, reverse=True)
labelsByWeight = np.array([k for (k, v) in labelsAndWeights])

length = labelsByWeight.size
truePositives = labelsByWeight.cumsum()
numPositive = truePositives[-1]
falsePositives = np.arange(1.0, length + 1, 1.) - truePositives

truePositiveRate = truePositives / numPositive
falsePositiveRate = falsePositives / (length - numPositive)

# Generate layout and plot data
fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(falsePositiveRate, truePositiveRate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
pass

display(fig)

# COMMAND ----------

# MAGIC %md ### **Part 5: Reduce feature dimension via feature hashing**

# COMMAND ----------

# MAGIC %md #### ** (5a) Hash function **
# MAGIC #### As we just saw, using a one-hot-encoding featurization can yield a model with good statistical accuracy.  However, the number of distinct categories across all features is quite large -- recall that we observed 233K categories in the training data.  Moreover, the full Kaggle training dataset includes more than 33M distinct categories, and the Kaggle dataset itself is just a small subset of Criteo's labeled data.  Hence, featurizing via a one-hot-encoding representation would lead to a very large feature vector. To reduce the dimensionality of the feature space, we will use feature hashing.

# COMMAND ----------

from collections import defaultdict
import hashlib

def hashFunction(numBuckets, rawFeats, printMapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use printMapping=True for debug purposes and to better understand how the hashing works.

    Args:
        numBuckets (int): Number of buckets to use as features.
        rawFeats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        printMapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)

# Reminder of the sample values:
# sampleOne = [(0, 'mouse'), (1, 'black')]
# sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
# sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]

# COMMAND ----------

# Use four buckets
sampOneFourBuckets = hashFunction(4, sampleOne, True)
sampTwoFourBuckets = hashFunction(4, sampleTwo, True)
sampThreeFourBuckets = hashFunction(4, sampleThree, True)

# Use one hundred buckets
sampOneHundredBuckets = hashFunction(100, sampleOne, True)
sampTwoHundredBuckets = hashFunction(100, sampleTwo, True)
sampThreeHundredBuckets = hashFunction(100, sampleThree, True)

print '\t\t 4 Buckets \t\t\t 100 Buckets'
print 'SampleOne:\t {0}\t\t {1}'.format(sampOneFourBuckets, sampOneHundredBuckets)
print 'SampleTwo:\t {0}\t\t {1}'.format(sampTwoFourBuckets, sampTwoHundredBuckets)
print 'SampleThree:\t {0}\t {1}'.format(sampThreeFourBuckets, sampThreeHundredBuckets)

# COMMAND ----------

# MAGIC %md #### ** (5b) Creating hashed features **
# MAGIC #### Next we will use this hash function to create hashed features for our CTR datasets.

# COMMAND ----------

def parseHashPoint(point, numBuckets):
    """Create a LabeledPoint for this observation using hashing.

    Args:
        point (str): A comma separated string where the first value is the label and the rest are
            features.
        numBuckets: The number of buckets to hash to.

    Returns:
        LabeledPoint: A LabeledPoint with a label (0.0 or 1.0) and a SparseVector of hashed
            features.
    """
    
    list_features_all = point.split(",")
    list_final = []
    n = len(list_features_all)
    for i in range(1,n):
      list_final.append((i-1,list_features_all[i]))
   
    sp = SparseVector(numBuckets,hashFunction(numBuckets, list_final, True))
    
    if (list_features_all[0] == "0"):
      lbp = LabeledPoint(0,sp)
    else:
      lbp = LabeledPoint(1,sp)
  
    return lbp

numBucketsCTR = 2 ** 15
hashTrainData = rawTrainData.map(lambda x: parseHashPoint(x, numBucketsCTR))
hashTrainData.cache()
hashValidationData = rawValidationData.map(lambda x: parseHashPoint(x, numBucketsCTR))
hashValidationData.cache()
hashTestData = rawTestData.map(lambda x: parseHashPoint(x, numBucketsCTR))
hashTestData.cache()

print hashTrainData.take(1)

# COMMAND ----------

# MAGIC %md #### ** (5c) Sparsity **
# MAGIC #### Since we have 33K hashed features versus 233K OHE features, we should expect OHE features to be sparser. Verify this hypothesis by computing the average sparsity of the OHE and the hashed training datasets.

# COMMAND ----------

def computeSparsity(data, d, n):
    """Calculates the average sparsity for the features in an RDD of LabeledPoints.

    Args:
        data (RDD of LabeledPoint): The LabeledPoints to use in the sparsity calculation.
        d (int): The total number of features.
        n (int): The number of observations in the RDD.

    Returns:
        float: The average of the ratio of features in a point to total features.
    """
    
    return data.map(lambda x: len(x.features.values)).mean()

averageSparsityHash = computeSparsity(hashTrainData, numBucketsCTR, nTrain)
averageSparsityOHE = computeSparsity(OHETrainData, numCtrOHEFeats, nTrain)

print 'Average OHE Sparsity: {0:.7e}'.format(averageSparsityOHE)
print 'Average Hash Sparsity: {0:.7e}'.format(averageSparsityHash)

# COMMAND ----------

# MAGIC %md #### ** (5d) Logistic model with hashed features **
# MAGIC #### Now let's train a logistic regression model using the hashed features.

# COMMAND ----------

numIters = 500
regType = 'l2'
includeIntercept = True

# Initialize variables using values from initial model training
bestModel = None
bestLogLoss = 1e10

# COMMAND ----------

stepSizes = [1,10]
regParams = [1e-6,1e-3]
for stepSize in stepSizes:
    for regParam in regParams:
        model = (LogisticRegressionWithSGD
                 .train(hashTrainData, numIters, stepSize, regParam=regParam, regType=regType,
                        intercept=includeIntercept))
        logLossVa = evaluateResults(model, hashValidationData)
        print ('\tstepSize = {0:.1f}, regParam = {1:.0e}: logloss = {2:.3f}'
               .format(stepSize, regParam, logLossVa))
        if (logLossVa < bestLogLoss):
            bestModel = model
            bestLogLoss = logLossVa

print ('Hashed Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, bestLogLoss))

# COMMAND ----------

# MAGIC %md #### **Visualization 3: Hyperparameter heat map**
# MAGIC #### We will now perform a visualization of an extensive hyperparameter search.  Specifically, we will create a heat map where the brighter colors correspond to lower values of `logLoss`.
# MAGIC #### The search was run using six step sizes and six values for regularization, which required the training of thirty-six separate models.  We have included the results below, but omitted the actual search to save time.

# COMMAND ----------

from matplotlib.colors import LinearSegmentedColormap

# Saved parameters and results.  Eliminate the time required to run 36 models
stepSizes = [3, 6, 9, 12, 15, 18]
regParams = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
logLoss = np.array([[ 0.45808431,  0.45808493,  0.45809113,  0.45815333,  0.45879221,  0.46556321],
                    [ 0.45188196,  0.45188306,  0.4518941,   0.4520051,   0.45316284,  0.46396068],
                    [ 0.44886478,  0.44886613,  0.44887974,  0.44902096,  0.4505614,   0.46371153],
                    [ 0.44706645,  0.4470698,   0.44708102,  0.44724251,  0.44905525,  0.46366507],
                    [ 0.44588848,  0.44589365,  0.44590568,  0.44606631,  0.44807106,  0.46365589],
                    [ 0.44508948,  0.44509474,  0.44510274,  0.44525007,  0.44738317,  0.46365405]])

numRows, numCols = len(stepSizes), len(regParams)
logLoss = np.array(logLoss)
logLoss.shape = (numRows, numCols)

fig, ax = preparePlot(np.arange(0, numCols, 1), np.arange(0, numRows, 1), figsize=(8, 7),
                      hideLabels=True, gridWidth=0.)
ax.set_xticklabels(regParams), ax.set_yticklabels(stepSizes)
ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Step Size')

colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
image = plt.imshow(logLoss,interpolation='nearest', aspect='auto',
                    cmap = colors)
pass

display(fig)

# COMMAND ----------

# MAGIC %md #### ** (5e) Evaluate on the test set **

# COMMAND ----------

# Log loss for the best model from (5d)
logLossTest = evaluateResults(bestModel, hashTestData)

# Log loss for the baseline model
logLossTestBaseline = hashTestData.map(lambda x: computeLogLoss(classOneFracTrain,x.label) ).mean()

print ('Hashed Features Test Log Loss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossTestBaseline, logLossTest))
