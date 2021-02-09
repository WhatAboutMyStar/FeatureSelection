# FeatureSelection
Feature selection Library Based on small sample in data mining

### Introduction
In the task of data mining, we often want to know which features in the data have an impact on the prediction target, and which features are used to improve the evaluation index most.<br>
Existing feature selection packages can only select features mechanically, but they can't tell us which features are most useful. as sklearn.feature_ Selection, although it can efficiently filter features, the converted features are numpy arrays, which lose the saved feature names of panda's dataframe. It is very difficult for us to find the feature names according to the data, especially in the case of a large number of features, it is difficult to determine the corresponding columns of the table data.<br>
For this purpose, we develop a feature filter Library Based on pandas dataframe table data mining, which can not only filter features efficiently, but also observe which features are filtered clearly and intuitively.

### how to use
Using the interface is similar to sklearn, using fit, fit_transform, transform method to filter features<br>
Using feature_importances_, columns_, keys (), etc to view filtered features
```
from featureSelection.tree import RandomForestSelection, DecisionTreeSelection
from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
x_data = boston.data
y_data = boston.target

x_data = pd.DataFrame(x_data)

dt = DecisionTreeSelection(top_k=5)
new_data = dt.fit_transform(x_data, y_data)
print(dt.feature_importances_)
print(dt.keys())
```

- Note: the input is the dataframe table data of pandas
