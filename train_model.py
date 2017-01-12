import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.externals import joblib

data = pd.read_csv("data/cs-training.csv")

# Data cleaning
data.dropna(inplace=True)
data.drop('Unnamed: 0', axis=1, inplace=True)
y = data['SeriousDlqin2yrs']
data.drop('SeriousDlqin2yrs', axis=1, inplace=True)
X = data

# Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=7)

# Creating the classifier
clf = LogisticRegression(class_weight="balanced")

# Training
clf.fit(X_train, y_train)

# Assessing the model

print "The confusion matrix is:"

y_pred = clf.predict(X_test)

print confusion_matrix(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

print "The Area under the curve (AUC) is:"

print auc(fpr, tpr)

# Saving the classifier

joblib.dump(clf, 'classifier.pkl')

print "The classifier was saved with success!!!!"











