from numpy.testing import assert_array_equal
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

from tf_rf.tfrf import SupervisedTermWeights


# Let's do language guessing.
docs = ["an apple a day keeps the doctor away",
        "time flies like an arrow",
        "the more the merrier",
        "the quick brown fox jumps over the lazy dog",
        "quod licet Iovi non licet bovi",
        "ut desint vires, tamen laudanda est voluntas",
        "gallia est omnis divisa in partes tres",
        "ceterum censeo carthaginem delendam esse",
        ]
y = ["en", "en", "en", "fr", "la", "la", "la", "fr"]

v = CountVectorizer(ngram_range=(1, 1))
X = v.fit_transform(docs)
# print(v.get_feature_names())
# print(len(v.get_feature_names()))


def supervised_term_weights():
    X_a = X.toarray()
    for weighting in SupervisedTermWeights._WEIGHTING:
    # for weighting in ['rf']:
        # for reduction in SupervisedTermWeights._REDUCE:
        for reduction in ['max', None]:
            sup_tw = SupervisedTermWeights(weighting=weighting,
                                           reduce=reduction)
            X1 = sup_tw.fit_transform(X, y)
            print(X1.shape)
            # X2 = sup_tw.fit(X, y).transform(X)

            X1_a = X1
            if sup_tw.reduce is not None:
                X1_a = X1.toarray()
                assert_array_equal(X1_a, X_a * sup_tw.weights_.T)
            else:
                assert_array_equal(X1_a, np.expand_dims(X_a, -1) * sup_tw.weights_.T)


if __name__ == '__main__':
    supervised_term_weights()
