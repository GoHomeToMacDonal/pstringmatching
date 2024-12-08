import math

import py_stringmatching as sm

import pstringmatching


def test_fodors_zagats(fodors, zagats):
    x = [f["name"] for f in fodors]
    y = [z["name"] for z in zagats]

    sim = pstringmatching.pairwise_bag_distance(x, y)
    measure = sm.BagDistance()

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            assert math.fabs(sim[i, j] - measure.get_sim_score(xi, yj)) <= 1e-5
