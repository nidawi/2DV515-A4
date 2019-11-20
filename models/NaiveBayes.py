from typing import List, Tuple, Dict
from lib.utils import mean, stdev, probability, probability_for_row
from timeit import default_timer as timer
from math import exp

class NaiveBayes:
  def __init__(self):
    self.__time = 0
    self.__predictTime = 0
    self.__data = []
    self.__labels = []
    self.__classMap = {}
    self.__summaryMap = {}
    self.__probabilityMap = {}

  def get_fit_time(self) -> float:
    return round(self.__time, 3)

  def get_predict_time(self) -> float:
    return round(self.__predictTime, 3)

  def get_class_count(self) -> int:
    return len(self.__summaryMap)

  def fit(self, X: List[List[float]], y: List[int]) -> None:
    """
    Trains this model on the input examples X and labels y.

    Partly based on: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
    """
    self.__time = timer()
    self.__data = X
    self.__labels = y

    # To train our model we need to:
    # 1. divide our dataset into their respective categories / classes
    self.__classMap = self.__separate_by_class()

    # 2. Calculate the mean & standard deviation values of each category and column combination
    # this method returns a dict with class names as keys
    # and a list of tuples (each tuple containing a column's mean, stdev, and length) as values.
    self.__summaryMap = self.__summarize_by_class(self.__classMap)

    # Done training.
    self.__time = (timer() - self.__time)

  def predict(self, X: List[List[float]]) -> List[int]:
    """
    Predicts the best-fitting class for each value in the input list X.

    Returns a list of suggested classes based on current training.
    """
    self.__predictTime = timer()
    predications = []

    for row in X:
      bestClass = self.___find_best_class(row)
      predications.append(bestClass)

    self.__predictTime = (timer() - self.__predictTime)
    return predications

  def predict_one(self, x: List[float]) -> int:
    """
    Predicts the best-fitting class for the provided values in x.

    Returns the class identifier number of the recommended class.
    """
    return self.___find_best_class(x)

  def ___find_best_class(self, row: List[float]) -> int:
    """
    Finds the best class for the provided values in row.

    Returns the class identifier number of the recommended class.
    """
    probabilities = self.__calculate_class_probabilities(row)
    bestMatch = max(probabilities.items(), key=lambda item:item[1])

    return bestMatch[0]

  def __calculate_class_probabilities(self, row: List[float]) -> dict:
    """
    Calculates class probabilities using Gaussian PDF. Values are transformed using the natural logarithm
    and then combined back in order to prevent numerical underflow problems. Returns a dict with class names / labels
    as keys and their respective probability as values.

    See this for more info on how the PDF is calculated:
    >>> lib.utils.probability
    >>> lib.utils.probability_for_row

    This is how the output could look (notice that class 0 is predicted due to highest probability):
    >>> dict { 0: 0.102, 1: 0.898 }
    """
    initProbabilities = { }

    for className, summaries in self.__summaryMap.items():
      # calculate probability using util wrapper
      initProbabilities[className] = probability_for_row(row, summaries)

    # normalize probabilities and add into a new dict (to maintain integrity of initial probabilities)
    probabilities = { }

    for className, prob in initProbabilities.items():
      # normalize by dividing by the sum of all probabilities
      total = sum(num for num in list(initProbabilities.values()))
      probabilities[className] = prob / total
    
    return probabilities

  def __separate_by_class(self) -> dict:
    """
    Separates a dataset based on its classes/labels.
    Returns a dict with class names / labels as keys and a list
    containing lists of row values in column-based order as values.

    :Example:

    This is the input:
    >>> | Column1 | Column2 | Class |
    >>> |    1    |    2    |   0   |
    >>> |    3    |    4    |   0   |
    >>> |    5    |    6    |   1   |

    This is the result:
    >>> dict { 0: [ [ 1, 2 ], [ 3, 4 ] ], 1: [ [ 5, 6 ] ] }
    """
    classMap = {}

    for i in range(len(self.__data)):
      label = self.__labels[i]

      if label not in classMap:
        classMap[label] = []

      classMap[label].append(self.__data[i])

    return classMap

  def __summarize_by_class(self, classMap: dict) -> dict:
    """
    Summarizes a dict containing a class-separated dataset.
    Returns a dict with class names as keys and a list of tuples describing summarized columns as values.
    
    Each tuple describes the values of one column and contains the following information:
    >>> (mean: float, standard deviation: float)

    The example below describes a dataset with two classes (= two keys) and two columns (= two tuples).
    >>> { 0: [ (1, 2), (2, 3) ], 1: [ (4, 5), (5, 6) ] }
    """
    summaryMap = {}

    for cName, rows in classMap.items():
      summaryMap[cName] = self.__summarize(rows)

    return summaryMap

  def __summarize(self, rows: List[List[float]]) -> Tuple[float, float]:
    """
    Returns a tuple representing the summary of every column belonging to the provided set of rows.
    The tuple contains the mean and the standard deviation. Like so:
    
    >>> (mean(column), stdev(column)) => (4, 2) # as an example

    The method uses the zip() function to retrieve column values from each row:

    >>> zip([1, 2, 3], [4, 5, 6]) # gives tupes containing (1, 4), (2, 5), and (3, 6)

    Since the input is a list of rows, you can, using the *-operator (similar to spread) create this:

    >>> zip(*[[1, 2], [3, 4]]) = zip([1, 2], [3, 4]) => (1, 3), (2, 4)
    """

    return [(mean(column), stdev(column)) for column in zip(*rows)]