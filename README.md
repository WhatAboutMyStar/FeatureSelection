# FeatureSelection
基于小样本数据挖掘的特征筛选库
[README_EngLish_Version](https://github.com/WhatAboutMyStar/FeatureSelection/blob/main/README-EN.md)
### 功能目的
在数据挖掘的任务中，我们往往想知道数据中的哪些特征对预测的目标有影响，
以及使用哪些特征在评价指标上有最大的提升。<br>
现有的特征选择包只能机械的选择特征，却不能直观的告知我们哪些特征最有用。
如sklearn.feature_selection，尽管它能高效的筛选特征，
但是转化出来的特征都是numpy数组，丢失了pandas的DataFrame存下来的特征名称，
我们要根据数据寻找特征名称非常困难，尤其特征数量众多的情况下，
我们很难确定表格数据对应的列。<br>
基于此目的，我们开发了一个基于pandas的DataFrame表格数据挖掘的特征筛选库，
既能够高效筛选特征，又能够清晰直观的观察到筛选出来的是哪些特征。

### 使用方法
使用接口和sklearn类似，用fit, fit_transform, transform方法筛选特征<br>
使用feature_importances_, columns_, keys()等查看筛选出来的特征
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
- 强调：输入是pandas的DataFrame表格数据