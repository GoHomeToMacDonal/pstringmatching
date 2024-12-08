import math

import py_stringmatching as sm

import pstringmatching
from tests.pstringmatching.fixture.fodors_zagats import fodors, zagats


def test_fodors_zagats(fodors, zagats):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    metadata = [
        (sm.QgramTokenizer(qval=1).tokenize, pstringmatching.pairwise_unigram_jaccard),
        (sm.QgramTokenizer(qval=2).tokenize, pstringmatching.pairwise_bigram_jaccard),
        (sm.QgramTokenizer(qval=3).tokenize, pstringmatching.pairwise_trigram_jaccard),
        (sm.WhitespaceTokenizer().tokenize, pstringmatching.pairwise_jaccard),
        (
            sm.AlphabeticTokenizer().tokenize,
            pstringmatching.pairwise_alphabetic_jaccard,
        ),
        (
            sm.AlphanumericTokenizer().tokenize,
            pstringmatching.pairwise_alphanumeric_jaccard,
        ),
    ]

    for tokenize, func in metadata:
        sim = func(x, y)
        measure = sm.Jaccard()

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                assert (
                    math.fabs(
                        sim[i, j] - measure.get_raw_score(tokenize(xi), tokenize(yj))
                    )
                    <= 1e-5
                )
