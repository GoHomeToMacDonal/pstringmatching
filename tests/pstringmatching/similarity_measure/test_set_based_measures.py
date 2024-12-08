import math

import py_stringmatching as sm

import pstringmatching

tokenizers = {
    'unigram': sm.QgramTokenizer(qval=1).tokenize,
    'bigram': sm.QgramTokenizer(qval=2).tokenize,
    'trigram': sm.QgramTokenizer(qval=3).tokenize,
    '': sm.WhitespaceTokenizer().tokenize,
    'alphabetic': sm.AlphabeticTokenizer().tokenize,
    'alphanumeric': sm.AlphanumericTokenizer().tokenize,
}


def do_test(fodors, zagats, func, measure):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    for name, tokenize in tokenizers.items():
        if name == '':
            name = '_'
        else:
            name = f'_{name}_'
        sim = getattr(pstringmatching, f'pairwise{name}{func}')(x, y)

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                assert (
                    math.fabs(
                        sim[i, j] - measure.get_raw_score(tokenize(xi), tokenize(yj))
                    )
                    <= 1e-5
                )


def test_jaccard(fodors, zagats):
    do_test(fodors, zagats, 'jaccard', sm.Jaccard())


def test_dice(fodors, zagats):
    do_test(fodors, zagats, 'dice', sm.Dice())


def test_cosine(fodors, zagats):
    do_test(fodors, zagats, 'cosine', sm.Cosine())


def test_overlap_coefficient(fodors, zagats):
    do_test(fodors, zagats, 'overlap_coefficient', sm.OverlapCoefficient())

