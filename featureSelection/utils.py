import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

def oneHotEncoder(row_data, x_data):
    """
    输入都是pandas的DataFrame表格
    :param row_data:原始数据
    :param x_data: 需要进行onehot处理的数据
    :return: 返回原始数据和onehot数据的表格
    """
    row_data = row_data.drop(columns=x_data.columns)
    new_data = pd.get_dummies(x_data)
    new_data = pd.concat([row_data, new_data], axis=1)
    return new_data

def init(x_data, y_data, std=0, test_size=0.1, random_state=666):
    """
    :param x_data: 输入二维数组
    :param y_data: 输入的预测标签
    :param std: 1是sklearn中的MinMaxScaler
                2是sklearn中的StandardScaler
                3是sklearn中的Normalizer
    :param test_size: 划分多少测试集， 默认是把所有数据划分为0.9的训练集和0.1的测试集
    :param random_state: 随机种子
    :return: 标准化归一化后的训练集和测试集
    """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    if std == 1:
        std = MinMaxScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)
    elif std == 2:
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)
    elif std == 3:
        std = Normalizer()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)
    return x_train, x_test, y_train, y_test

class FeatureSelection:
    def __init__(self):
        self.columns_ = None
        self.best_score_ = None
        self._index = None
        self._n_jobs = -1

    def fit(self, x_data):
        return self

    def fit_transform(self, x_data):
        self.fit(x_data)
        return self.transform(x_data)

    def transform(self, x_data):
        return x_data.iloc[:, self._index]

    def keys(self):
        return self.columns_

    def setNjobs(self, n_jobs):
        self._n_jobs = n_jobs

