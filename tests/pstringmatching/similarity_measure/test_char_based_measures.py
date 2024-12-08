import math

import py_stringmatching as sm

import pstringmatching


def do_test(fodors, zagats, func, measure):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    sim = func(x, y)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            assert math.fabs(sim[i, j] - measure.get_sim_score(xi, yj)) <= 1e-5


def test_bag_distance(fodors, zagats):
    do_test(
        fodors, zagats, pstringmatching.pairwise_bag_distance, measure=sm.BagDistance()
    )


def test_jaro(fodors, zagats):
    do_test(fodors, zagats, pstringmatching.pairwise_jaro, measure=sm.Jaro())


def test_jaro_winkler(fodors, zagats):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    for weight in [0.1, 0.2, 0.3]:
        sim = pstringmatching.pairwise_jaro_winkler(x, y, weight)
        measure = sm.JaroWinkler(weight)
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                assert math.fabs(sim[i, j] - measure.get_sim_score(xi, yj)) <= 1e-5


def test_levenshtein(fodors, zagats):
    do_test(
        fodors, zagats, pstringmatching.pairwise_levenshtein, measure=sm.Levenshtein()
    )


def test_pairwise_needleman_wunsch(fodors, zagats):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    sim = pstringmatching.pairwise_needleman_wunsch(x, y)
    measure = sm.NeedlemanWunsch()
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            assert math.fabs(sim[i, j] - measure.get_raw_score(xi, yj)) <= 1e-5


def test_pairwise_smith_waterman(fodors, zagats):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    sim = pstringmatching.pairwise_smith_waterman(x, y)
    measure = sm.SmithWaterman()
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            assert math.fabs(sim[i, j] - measure.get_raw_score(xi, yj)) <= 1e-5
