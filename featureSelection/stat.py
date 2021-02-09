from featureSelection.utils import FeatureSelection

class VarianceThreshold(FeatureSelection):
    """
    通过特征的方差来选取特征，需要设定一个阈值，方差小于阈值则丢弃该特征，方差大于阈值则留下特征
    输入是一个pandas的DataFrame，此举目的是为了保留特征的名称，而不是只剩下数字，挖掘数据还要求寻求一个可解释性，能方便的知道具体特征的名字至关重要

    Examples
    --------
    >>> var = VarianceThreshold(threshold=0.1)
    >>> var.fit(x_data)
    >>> new_data = var.transform(x_data)
    >>> var.keys()
    """
    def __init__(self, threshold=0.1):
        super(VarianceThreshold, self).__init__()
        self._threshold = threshold

    def fit(self, x_data):
        col = x_data.shape[1]
        index = []
        for i in range(col):
            if x_data.iloc[:, i].var() > self._threshold:
                index.append(i)
        self.columns_ = x_data.iloc[0, index].keys()
        self._index = index.copy()

        return self

    def setThreshold(self, threshold):
        self._threshold = threshold

    def __repr__(self):
        return "VarianceThreshold(threshold={})".format(self._threshold)




if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import pandas as pd
    boston = load_boston()
    x_data = boston.data
    y_data = boston.target
    x_data = pd.DataFrame(x_data)

    var = VarianceThreshold(1)
    var.fit(x_data)
    new_data = var.transform(x_data)
    print(x_data.var())
    print(var.keys())