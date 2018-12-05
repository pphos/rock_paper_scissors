# -*- coding: utf-8 -*-


class Bunch(dict):
    """
    scikit-learn.utils.Bunch class
    URL:https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py

    データセット用のコンテナオブジェクト
    ex)
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.a = 3
    """

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise ArithmeticError(key)
