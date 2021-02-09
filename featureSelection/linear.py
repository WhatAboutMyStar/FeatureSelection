import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from featureSelection.utils import FeatureSelection


class LassoSelecion(FeatureSelection):
    """
    使用Lasso回归进行特征选择, 输入是pandas的DataFrame，此举目的是为了保留特征的名称，而不是只剩下数字，
    挖掘数据还要求寻求一个可解释性，能方便的知道具体特征的名字至关重要

    Examples
    --------
    >>> lr = LassoSelection()
    >>> lr.fit(x_data, y_data)
    >>> new_data = lr.transform(x_data)
    >>> lr.coef_
    >>> lr.alpha_
    >>> lr.keys()
    """

    def __init__(self, threshold=5):
        """
        :param threshold: 超参数alpha的阈值上限
                          我们的搜索方式是从0.01一直搜索到threshold的值，每次移动0.01的步长。
                          threshold默认是5
        """
        super(LassoSelecion, self).__init__()
        self.coef_ = None
        self.alpha_ = None
        self._threshold = threshold

    def fit(self, x_data, y_data):

        self._search(x_data, y_data)
        lr = Lasso(self.alpha_, max_iter=10000)
        lr.fit(x_data, y_data)
        self.coef_ = lr.coef_

        select = (lr.coef_ == 0)            #[0 0 1 1 0 0 1] 为1的部分是要被删除的特征
        ls = np.arange((x_data.shape[1]))   #[0 1 2 3 4 5 6]
        select = select * ls                #[0 0 2 3 0 0 6] 为0的部分是留下来的特征

        index = []
        if lr.coef_[0] != 0:                #第一个数索引本身无论如何都是0，需要特殊处理
            index.append(0)
        for i in range(1, len(select)):
            if select[i] == 0:
                index.append(i)

        self.columns_ = x_data.iloc[0, index].keys()
        self._index = index.copy()
        return self

    def fit_transform(self, x_data, y_data):
        self.fit(x_data, y_data)
        return self.transform(x_data)

    def _search(self, x_data, y_data):
        threshold = np.arange(0.01, self._threshold, 0.01)
        parameters = {'alpha': threshold}
        lr = Lasso(max_iter=10000)
        gs = GridSearchCV(lr, parameters, n_jobs=self._n_jobs)
        gs.fit(x_data, y_data)
        self.best_score_ = gs.best_score_
        # self.coef_ = lr.coef_
        self.alpha_ = gs.best_params_['alpha']

    def __repr__(self):
        return "LassoSelection(threshold={})".format(self._threshold)

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import pandas as pd
    boston = load_boston()
    x_data = boston.data
    y_data = boston.target
    x_data = pd.DataFrame(x_data)

    lr = LassoSelecion()
    lr.fit(x_data, y_data)
    print(lr.coef_)
    print(lr.keys())
    print(lr.alpha_)
    print(lr.best_score_)