from models.Dataset import Dataset
from models.NaiveBayes import NaiveBayes
from models.CrossValidation import crossval_predict, crossval_predict_evaluation
from lib.utils import accuracy_score, confusion_matrix, present_matrix, present_accuracies

BANKNOTE_FILE = "data/banknote_authentication.csv"
IRIS_FILE = "data/iris.csv"
CROSS_VAL_FOLDS = 5

# parse selected file
fdb = Dataset(IRIS_FILE)

def run_naive_bayes_example():
  print("Naive Bayes @ training data")

  # create and train model using the previously loaded file
  model = NaiveBayes()
  model.fit(fdb.get_data(), fdb.get_labels())
  print("Naive Bayes model was trained with data from %s in %s seconds." % (fdb.get_file_name(), model.get_fit_time()))

  # run predictions
  predictions = model.predict(fdb.get_data())
  actuals = fdb.get_labels()

  print("Predictions complete after %s seconds." % model.get_predict_time())

  # calculate accuracy
  accuracy = accuracy_score(predictions, actuals)

  print("Accuracy: %s%s (%d/%d correctly classified)"
  % (accuracy["accuracy"], "%", accuracy["matching"], accuracy["total"]))

  # produce a matrix
  matrix = confusion_matrix(predictions, actuals)

  # present matrix
  present_matrix(matrix)

def run_cross_val_example():
  print("Cross Validation (%s), %s-fold" % (fdb.get_file_name(), CROSS_VAL_FOLDS))

  # calculate cross validation
  predictions = crossval_predict(fdb.get_data(), fdb.get_labels(), CROSS_VAL_FOLDS)
  actuals = fdb.get_labels()

  # calculate accuracy
  accuracy = accuracy_score(predictions, actuals)

  print("Accuracy: %s%s (%d/%d correctly classified)"
  % (accuracy["accuracy"], "%", accuracy["matching"], accuracy["total"]))

  # produce a matrix
  matrix = confusion_matrix(predictions, actuals)

  # present matrix
  present_matrix(matrix)

def run_cross_val_eval_example():
  print("Cross Validation (%s), %s-fold" % (fdb.get_file_name(), CROSS_VAL_FOLDS))

  # calculate fold-based accuracies for cross-validation
  accuracies = crossval_predict_evaluation(fdb.get_data(), fdb.get_labels(), CROSS_VAL_FOLDS)

  # present totals
  totalAccuracy = (sum(acc["accuracy"] for acc in accuracies)) / CROSS_VAL_FOLDS
  totalExamples = (sum(acc["total"] for acc in accuracies))
  totalMatching = (sum(acc["matching"] for acc in accuracies))
  
  print("Accuracy: %s%s (%d/%d correctly classified)"
  % (round(totalAccuracy, 2), "%", totalMatching, totalExamples))

  # present accuracies as a table
  present_accuracies(accuracies)

#run_naive_bayes_example()

#run_cross_val_example()

#run_cross_val_eval_example()