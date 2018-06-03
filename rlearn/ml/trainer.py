import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def _feature_construction(features):
    """Construct additional features"""
    new_features = features
    return new_features


def train_model(X, y, model_type='regression', task_type='tree'):
    if model_type == 'regression':
        if task_type == 'tree':
            clf = DecisionTreeRegressor()
        else:
            clf = RandomForestRegressor()
    else:
        if task_type == 'tree':
            clf = DecisionTreeClassifier()
        else:
            clf = RandomForestClassifier()

    clf = clf.fit(X, y)

    return clf


def score(data: list, clf):
    """Return scores for data """
    print(data)
    return clf.predict(data)


def test_train():
    X = [(2.0, 1.0),
         (2.0, 4.0)]

    y = [0.5, 2.5]
    clf = train_model(X, y)

    results = clf.predict(X)

    assert sum(y) == sum(results)


if __name__ == '__main__':
    test_train()
