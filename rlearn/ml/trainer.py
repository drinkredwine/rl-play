import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import sklearn


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
    from pprint import pprint
    pprint(clf.tree_.__getstate__())

    assert sum(y) == sum(results)


def tree_to_json(decision_tree, feature_names=None):
    from warnings import warn

    js = ""

    def node_to_str(tree, node_id, criterion):
        if not isinstance(criterion, sklearn.tree.tree.six.string_types):
            criterion = "impurity"

        value = tree.value[node_id]
        if tree.n_outputs == 1:
            value = value[0, :]

        jsonValue = ', '.join([str(x) for x in value])

        if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            return '"id": "%s", "criterion": "%s", "impurity": "%s", "samples": "%s", "value": [%s]' \
                   % (node_id,
                      criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id],
                      jsonValue)
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = tree.feature[node_id]

            if "=" in feature:
                ruleType = "="
                ruleValue = "false"
            else:
                ruleType = "<="
                ruleValue = "%.4f" % tree.threshold[node_id]

            return '"id": "%s", "rule": "%s %s %s", "%s": "%s", "samples": "%s"' \
                   % (node_id,
                      feature,
                      ruleType,
                      ruleValue,
                      criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id])

    def recurse(tree, node_id, criterion, parent=None, depth=0):
        tabs = "  " * depth
        js = ""

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        js = js + "\n" + \
             tabs + "{\n" + \
             tabs + "  " + node_to_str(tree, node_id, criterion)

        if left_child != sklearn.tree._tree.TREE_LEAF:
            js = js + ",\n" + \
                 tabs + '  "left": ' + \
                 recurse(tree, \
                         left_child, \
                         criterion=criterion, \
                         parent=node_id, \
                         depth=depth + 1) + ",\n" + \
                 tabs + '  "right": ' + \
                 recurse(tree, \
                         right_child, \
                         criterion=criterion, \
                         parent=node_id,
                         depth=depth + 1)

        js = js + tabs + "\n" + \
             tabs + "}"

        return js

    if isinstance(decision_tree, sklearn.tree.tree.Tree):
        js = js + recurse(decision_tree, 0, criterion="impurity")
    else:
        js = js + recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

    return js


if __name__ == '__main__':
    test_train()
    X = [(2.0, 1.0),
         (2.0, 4.0)]

    y = [0.5, 2.5]
    clf = train_model(X, y)

    results = clf.predict(X)
    from pprint import pprint
    pprint(clf.tree_.__getstate__())

    js = tree_to_json(clf, ['f-x', 'f-y'])
    print(js)

