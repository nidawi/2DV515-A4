import csv
from typing import List

class Dataset:
  def __init__(self, file: str):
    self.__rawFile = file
    self.__headers = []
    self.__labelMap = {}
    self.__dataMap = {}
    self.__rawLabels = []
    self.__load_file()

  def get_file_name(self) -> str:
    return self.__rawFile

  def get_true_label_for(self, labelAlias: int) -> str:
    if labelAlias in self.__labelMap:
      return self.__labelMap[labelAlias]
    else:
      raise AttributeError()

  def get_labels(self) -> List[int]:
    return self.__rawLabels

  def get_data(self) -> List[List[float]]:
    result = []
    for data in self.__dataMap.values():
      result.extend(data)

    return result # rewrite as a sum()-operation?

  def __load_file(self):
    with open(self.__rawFile) as csvfile:
      csvreader = csv.reader(csvfile)
      self.__headers = next(csvreader) # load headers

      for row in csvreader: # load the rest
        label = row[-1]
        if label not in self.__labelMap:
          self.__labelMap[label] = len(self.__labelMap)

        data = list(map(lambda x : float(x), row[0:-1]))
        dataLabel = self.__labelMap[label]
        self.__rawLabels.append(dataLabel)
        if dataLabel not in self.__dataMap:
          self.__dataMap[dataLabel] = []
        
        self.__dataMap[dataLabel].append(data)
