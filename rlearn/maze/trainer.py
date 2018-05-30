import numpy as np
from sklearn import tree


def state_to_predictors(observation):
    """State to Features transformation and features construction

    (target, att1, att2, ..., attn)"""

    r = []
    for x in observation[1:]:
        if type(x) in [list, tuple]:
            for z in x:
                try:
                    r.append(float(z))
                except:
                    pass
        else:
            try:
                r.append(float(x))
            except:
                r.append(0.0)

    return np.array(r)


def train_model(memory: list):
    data = [state_to_predictors(x) for x in memory]

    X = np.array(data, dtype='float')
    y = list(map(lambda i: i[0], memory))

    # clf = RandomForestRegressor()
    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf = clf.fit(X, y)

    return clf


def score(memory: list, clf) -> float:
    for o in memory:
        oo = state_to_predictors(o)
        yield clf.predict([oo]), oo[0]


def test_state_to_predictors():
    assert list(state_to_predictors((9.0, 1.0, 2.0, 3.0))) == [1.0, 2.0, 3.0]
    assert list(state_to_predictors((9.0, (1.0, 1.1), 2.0, 3.0))) == [1.0, 1.1, 2.0, 3.0]


def test_train():
    memory = np.array([
        (0.5, 1, 2.0, 1.0),
        (2.5, 2, 2.0, 4.0)
    ], dtype='float')

    clf = train_model(memory)

    results = []

    for prediction in score(memory, clf):
        results.append(prediction)

    assert [(np.array([0.5]), 1.0), (np.array([2.5]), 2.0)] == results


if __name__ == '__main__':
    test_train()
