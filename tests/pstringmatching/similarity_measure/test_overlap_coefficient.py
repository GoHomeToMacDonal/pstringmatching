import math

import py_stringmatching as sm

import pstringmatching


def test_fodors_zagats(fodors, zagats):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    metadata = [
        (sm.QgramTokenizer(qval=1).tokenize, pstringmatching.pairwise_unigram_overlap_coefficient),
        (sm.QgramTokenizer(qval=2).tokenize, pstringmatching.pairwise_bigram_overlap_coefficient),
        (sm.QgramTokenizer(qval=3).tokenize, pstringmatching.pairwise_trigram_overlap_coefficient),
        (sm.WhitespaceTokenizer().tokenize, pstringmatching.pairwise_overlap_coefficient),
        (
            sm.AlphabeticTokenizer().tokenize,
            pstringmatching.pairwise_alphabetic_overlap_coefficient,
        ),
        (
            sm.AlphanumericTokenizer().tokenize,
            pstringmatching.pairwise_alphanumeric_overlap_coefficient,
        ),
    ]

    for tokenize, func in metadata:
        sim = func(x, y)
        measure = sm.OverlapCoefficient()

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                assert (
                    math.fabs(
                        sim[i, j] - measure.get_raw_score(tokenize(xi), tokenize(yj))
                    )
                    <= 1e-5
                )
