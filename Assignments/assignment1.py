"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import math
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass

    def thomas_solve(self, d):
        # Function to solve a tridiagonal linear system using Thomas algorithm.

        n_p = len(d)
        a = [1 for i in range(n_p - 1)]
        a[-1] = 2
        b = [4 for i in range(n_p)]
        b[0] = 2
        b[-1] = 7
        c = [1 for i in range(n_p - 1)]

        # Forward elimination
        for i in range(1, len(d)):
            temp = a[i - 1] / b[i - 1]
            b[i] = b[i] - temp * c[i - 1]
            d[i] = d[i] - temp * d[i - 1]

        x = b
        x[-1] = d[-1] / b[-1]

        # Backward substitution
        for j in range(len(d) - 2, -1, -1):
            x[j] = (d[j] - c[j] * x[j + 1]) / b[j]
        return x

    def get_bezier_coef(self, points):
        # Calculate Bezier coefficients from given points.

        n_p = len(points) - 1

        # Build points vector
        P = np.array([2 * (2 * points[i] + points[i + 1]) for i in range(n_p)])
        P[0] = points[0] + 2 * points[1]
        P[n_p - 1] = 8 * points[n_p - 1] + points[n_p]

        # Solve linear systems for A and B coefficients
        A = self.thomas_solve(P)
        B = [0] * n_p
        for i in range(n_p - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n_p - 1] = (A[n_p - 1] + points[n_p]) / 2

        # Return cubic Bezier functions for each segment
        return [
            self.get_cubic(points[i], A[i], B[i], points[i + 1])
            for i in range(len(points) - 1)
        ]

    # Returns the general Bezier cubic formula given 4 control points
    def get_cubic(self, a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,
                                                                                                          2) * c + np.power(
            t, 3) * d

    def get_points(self, f, a, b, n):
        # Generate points for interpolation.

        if n != 1:
            x_values = np.linspace(a, b, n)
            y_values = []
            for x in x_values:
                y_values.append(f(x))
            points = np.array(list(zip(x_values, y_values)))
        else:
            points = [[(b - a) / 2, f((b - a) / 2)]]
        return points

    def get_index(self, x, points):
        # Get the index of the interval containing x in the given points.

        index = 0
        for i in range(len(points) - 1):
            if points[i][0] <= x <= points[i + 1][0]:
                break
            index += 1
        return index

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        points = self.get_points(f, a, b, n)
        if n != 1:
            normal = points[1][0] - points[0][0]
            curves = self.get_bezier_coef(points)

        def g(x):
            # Interpolation function.

            if n != 1:
                idx = self.get_index(x, points)
                t = (x - points[idx][0]) / normal
                res = curves[idx](t)[1]
                return res
            else:
                return points[0][1]

        ans = lambda x: g(x)
        return ans


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 1)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
