from featureSelection.tree import RandomForestSelection, DecisionTreeSelection
from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
x_data = boston.data
y_data = boston.target
x_data = pd.DataFrame(x_data)
print(x_data.shape)
dt = DecisionTreeSelection(top_k=5)
print(dt)
new_data = dt.fit_transform(x_data, y_data)
print(new_data.shape)
print(dt.feature_importances_)
print(dt.columns_)

dt = RandomForestSelection(top_k=5)
new_data = dt.fit_transform(x_data, y_data)
print(new_data.shape)
print(dt.feature_importances_)
print(dt.keys())
print(dt)