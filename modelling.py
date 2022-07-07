import pandas as pd
from embedding import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier
import time
start = time.time()

print('Logistic Regression')
clf = LogisticRegression()
clf.fit(X_train,y_train)
predicted_labels = clf.predict(X_test)
print("The accuracy is {}".format(accuracy_score(y_test, predicted_labels)))
print(classification_report(y_test, predicted_labels))

print("XGBoost")
model = XGBClassifier()
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)
print("The accuracy is {}".format(accuracy_score(y_test, predicted_labels)))
print(classification_report(y_test, predicted_labels))

print(time.time() - start)