import numpy as np
from utils import FeatureSelection, init
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class BaseTree:
    def __init__(self,
                 max_depth,
                 max_features,
                 max_leaf_nodes,
                 top_k,
                 regression):
        self._max_depth = max_depth
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._top_k = top_k
        self._regression = regression
        self._model = None


class DecisionTreeSelection(FeatureSelection, BaseTree):
    def __init__(self,
                 top_k,
                 max_depth=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 regression=True):
        super(DecisionTreeSelection, self).__init__()
        BaseTree.__init__(self,
                          max_depth=max_depth,
                          max_features=max_features,
                          max_leaf_nodes=max_leaf_nodes,
                          top_k=top_k,
                          regression=regression)


    def fit(self, x_data, y_data):
        if self._regression:
            self._model = DecisionTreeRegressor(max_depth=self._max_depth,
                                                max_features=self._max_features,
                                                max_leaf_nodes=self._max_leaf_nodes)
        else:
            self._model = DecisionTreeClassifier(max_depth=self._max_depth,
                                                 max_features=self._max_features,
                                                 max_leaf_nodes=self._max_leaf_nodes)
        self._model.fit(x_data, y_data)
        self._index = np.argsort(-self._model.feature_importances_)[:self._top_k]
        self.columns_ = x_data.keys()[self._index]

        return self

    def fit_transform(self, x_data, y_data):
        self.fit(x_data, y_data)
        return self.transform(x_data)

    @property
    def feature_importances_(self):
        return dict(zip(self._model.feature_importances_[self._index], self.columns_))


    def __repr__(self):
        return "DecisionTreeSelection()"

class RandomForestSelection(FeatureSelection, BaseTree):
    def __init__(self,
                 top_k,
                 n_estimators=100,
                 max_depth=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 regression=True):
        super(RandomForestSelection, self).__init__()
        BaseTree.__init__(self,
                          max_depth=max_depth,
                          max_features=max_features,
                          max_leaf_nodes=max_leaf_nodes,
                          top_k=top_k,
                          regression=regression)
        self._n_estimators = n_estimators

    def fit(self, x_data, y_data):
        if self._regression:
            self._model = RandomForestRegressor(n_estimators=self._n_estimators,
                                                max_depth=self._max_depth,
                                                max_features=self._max_features,
                                                max_leaf_nodes=self._max_leaf_nodes,
                                                n_jobs=self._n_jobs)
        else:
            self._model = RandomForestClassifier(n_estimators=self._n_estimators,
                                                 max_depth=self._max_depth,
                                                 max_features=self._max_features,
                                                 max_leaf_nodes=self._max_leaf_nodes,
                                                 n_jobs=self._n_jobs)

        self._model.fit(x_data, y_data)
        self._index = np.argsort(-self._model.feature_importances_)[:self._top_k]
        self.columns_ = x_data.keys()[self._index]

        return self

    def fit_transform(self, x_data, y_data):
        self.fit(x_data, y_data)
        return self.transform(x_data)

    @property
    def feature_importances_(self):
        return dict(zip(self._model.feature_importances_[self._index], self.columns_))

    def __repr__(self):
        return "RandomForestSelection()"


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import pandas as pd
    boston = load_boston()
    x_data = boston.data
    y_data = boston.target
    x_data = pd.DataFrame(x_data)
    print(x_data.shape)
    dt = DecisionTreeSelection(top_k=5)
    new_data = dt.fit_transform(x_data, y_data)
    print(new_data.shape)
    print(dt.feature_importances_)
    print(dt.columns_)

    dt = RandomForestSelection(top_k=5)
    new_data = dt.fit_transform(x_data, y_data)
    print(new_data.shape)
    print(dt.feature_importances_)
    print(dt.keys())