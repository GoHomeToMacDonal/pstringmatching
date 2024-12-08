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
    do_test(
        fodors, zagats, pstringmatching.pairwise_jaro, measure=sm.Jaro()
    )

# def test_jaro_winkler(fodors, zagats):
#     do_test(
#         fodors, zagats, pstringmatching.pairwise_jaro_winkler, measure=sm.JaroWinkler()
#     )
