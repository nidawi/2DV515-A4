## Assignment Checklist
### E
1. ~~Implement the Naïve Bayes algorithm, using the provided code structure.~~
2. ~~Train the model on the Iris and Banknote authentication datasets.~~
3. ~~Calculate classification accuracies for both datasets.~~
### C-D
1. ~~Implement code for generating confusion matrices.~~
### A-B
1. ~~Implement code for n-fold cross-validation.~~
2. ~~It shall be possible to use 3, 5 or 10 folds.~~
3. ~~Calculate accuracy score for 5-fold cross-validation on both datasets.~~
### Code Structure Requirements
NaiveBayes (fit, predict) in models/NaiveBayes.py
accuracy_score in lib/util.py
confusion_matrix in lib/util.py
crossval_predict in models/CrossValidation.py
### Execution Examples
> Note: I am still a Python rookie.
#### Iris (naive bayes, all data)
![Iris (naive bayes, all data)](https://i.gyazo.com/b9b5724cad638598046a843ed14181f7.png)
#### Iris (cross-validation, 5-fold)
![Iris (cross-validation, 5-fold-1)](https://i.gyazo.com/9aed78fb0fbcce94e75aada99eadf905.png)
![Iris (cross-validation, 5-fold-2)](https://i.gyazo.com/2ed5d459c20b11e376c84204a5af0723.png)
#### Banknote (naive bayes, all data)
![Banknote (naive bayes, all data)](https://i.gyazo.com/0869598140d295b021e8e3c926bd0062.png)
#### Banknote (cross-validation, 5-fold)
![Banknote (cross-validation, 5-fold-1)](https://i.gyazo.com/65f7c246b9ab167a86f7d15738037047.png)
![Banknote (cross-validation, 5-fold-2)](https://i.gyazo.com/1ee18c17241a360fbc3ef046450716fc.png)