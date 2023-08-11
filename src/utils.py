import numpy as np
from scipy.optimize import minimize
from scipy.stats import rv_continuous


def Rotate(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def bezier_to_point_mindist(p0, p1, p2, q):

    def bezier_func(t, p0, p1, p2):
        a = p0 - 2 * p1 + p2
        b = p1 - p2
        c = p2
        return a * t ** 2 + 2 * b * t + c

    def fun(t, p0, p1, p2, q):
        return np.linalg.norm(bezier_func(t, p0, p1, p2) - q)

    res = minimize(fun, args=(p0, p1, p2, q), x0=np.array([0]), method="SLSQP",
                   constraints={"type": "ineq", "fun": lambda x: 0})
    dist = res.fun
    closest_point = bezier_func(res.x, p0, p1, p2)
    return closest_point, dist

class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)
