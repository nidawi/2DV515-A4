from models.NaiveBayes import NaiveBayes
from lib.utils import accuracy_score, confusion_matrix, present_matrix
from random import randrange
from typing import List

def crossval_predict(X: List[List[float]], y: List[int], folds: int) -> List[int]:
  """
  Runs n-fold cross-validation on the provided dataset, X, with the given labels, y.

  Returns a list of class predictions in true order.

  Format (output):
  >>> [ int, int, int ] # and so on

  Procedure:

  1. Shuffle the dataset randomly.
  2. Split the dataset into k groups
  3. For each unique group:
  3.1. Take the group as a hold out or test data set
  3.2. Take the remaining groups as a training data set
  3.3. Fit a model on the training set and evaluate it on the test set
  3.4. Retain the evaluation score and discard the model
  4. Summarize the skill of the model using the sample of model evaluation scores
  """
  # add correct labels to each row
  for i in range(len(X)):
    X[i].append(y[i])
    X[i].append(None) # placeholder for prediction

  # step 1 and 2
  workFolds = split_into_folds(X, folds)

  for fold in workFolds:
    # convert all other folds into training set: step 3.1 and 3.2
    trainingSet = list(filter(lambda x : x is not fold, workFolds))
    trainingSet = sum(trainingSet, []) # todo: figure out why this works
    trainingLabels = list(map(lambda x : x[-2], trainingSet)) # generate training set-specific labels
    trainingSet = list(map(lambda x : x[0:-2], trainingSet)) # remove labels etc. from the training set
    # otherwise the model will think that they represent another column.
      
    # setup the model
    model = NaiveBayes()
    model.fit(trainingSet, trainingLabels)

    # calculate predictions
    prediction = model.predict(fold)

    # add predictions to our main dataset
    for i in range(len(fold)):
      fold[i][-1] = prediction[i]

  # produce a true list of predictions
  preds = list(map(lambda x : x[-1], X))

  return preds

def crossval_predict_evaluation(X: List[List[float]], y: List[int], folds: int) -> List[dict]:
  """
  Runs n-fold cross-validation on the provided dataset, X, with the given labels, y.

  Returns a list containing accuracy results for each fold. Each accuracy results
  contains the following information: total examples in the fold, matching examples
  (correctly predicted), and accuracy in percent.

  Format (output):
  >>> [ { total: int, matching: int, accuracy: float } ]

  Procedure:

  1. Shuffle the dataset randomly.
  2. Split the dataset into k groups
  3. For each unique group:
  3.1. Take the group as a hold out or test data set
  3.2. Take the remaining groups as a training data set
  3.3. Fit a model on the training set and evaluate it on the test set
  3.4. Retain the evaluation score and discard the model
  4. Summarize the skill of the model using the sample of model evaluation scores
  """
  # add correct labels to each row
  for i in range(len(X)):
    X[i].append(y[i])

  # step 1 and 2
  workFolds = split_into_folds(X, folds)
  scores = []

  for fold in workFolds:
    # convert all other folds into training set: step 3.1 and 3.2
    trainingSet = list(filter(lambda x : x is not fold, workFolds))
    trainingSet = sum(trainingSet, []) # todo: figure out why this works
    trainingLabels = list(map(lambda x : x[-1], trainingSet)) # generate training set-specific labels
    trainingSet = list(map(lambda x : x[0:-1], trainingSet)) # remove labels etc. from the training set
    # otherwise the model will think that they represent another column.
      
    # setup the model
    model = NaiveBayes()
    model.fit(trainingSet, trainingLabels)

    # calculate predictions
    prediction = model.predict(fold)

     # fetch the actual position [-2] for comparison
    actual = [row[-1] for row in fold]

    # calculate accuracy for fold and add to scores
    acc = accuracy_score(prediction, actual)
    scores.append(acc)

  return scores

def split_into_folds(dataset: List[List[float]], foldCount: int) -> List[List[List[float]]]:
  """
  Splits a dataset into the specified number of folds.
  """
  folds = []
  foldSize = int(len(dataset) / foldCount)
  dataset = list(dataset) # copy the dataset

  for _ in range(foldCount):
    fold = []
    while len(fold) < foldSize:
      index = randrange(len(dataset))
      fold.append(dataset.pop(index))
    folds.append(fold)

  # if we get any extra examples that do not split evenly into any folds,
  # we try to divide them evenly across the folds (as well as we can)
  for i in range(len(dataset)):
    folds[i].append(dataset[i])

  return folds