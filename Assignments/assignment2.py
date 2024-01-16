"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable
import math

class Assignment2:

    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution
        # Initialize an empty list to store intersection points
        X = []

        # Function representing the difference between f1 and f2
        function = lambda x: f1(x) - f2(x)

        # Function to calculate the derivative of the difference function
        def derivative(x):
            h = 1e-10
            return (function(x + h) - function(x)) / h

        # Newton's method for root finding
        def newton(a, maxiter):
            try:
                n = 0
                x_i = a
                while abs(function(x_i)) >= maxerr:
                    z = derivative(x_i)
                    if z == 0.0:
                        return None
                    x_i = x_i - function(x_i) / z
                    n = n + 1
                    if n == maxiter:
                        return bisection(x_i, b)
                    if n == maxiter and function(x_i) / derivative(x_i) < 1e-10:
                        return x_i
                if n != maxiter and n > 0:
                    return x_i
                return bisection(x_i, b)
            except:
                return bisection(a, b)

        # Bisection method for root finding
        def bisection(a, b):
            try:
                n = 0
                x = (a + b) / 2
                while abs(function(x)) >= maxerr:
                    if function(x) == 0:
                        return x
                    if function(a) is None or function(x) is None:
                        return None
                    else:
                        if function(a) * function(x) < 0:
                            b = x
                        else:
                            a = x
                        x = (a + b) / 2
                    n = n + 1
                    if n > 100:
                        return x
                return x
            except:
                return None

        # Generate a range of points for initial guesses
        rang = np.linspace(a, b, 500)
        for x in rang:
            sol = newton(x, maxiter=50)
            if sol is not None:
                if sol not in X:
                    if a - maxerr <= sol <= b + maxerr:
                        if abs(function(sol)) < maxerr:
                            X.append(sol)

        # Process the obtained intersection points
        solution = []
        X.sort()
        for i in range(0, len(X) - 1):
            if X[i + 1] - X[i] > maxerr:
                solution.append(X[i])
            else:
                X[i + 1] = X[i]
        if len(X) > 0:
            solution.append(X[-1])

        return solution



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):
    #
    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    # def test_poly2(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1 = strong_oscilations()
    #     f2 = strong_oscilations2()
    #     X = ass2.intersections(f1, f2, -3, 3, maxerr=0.001)
    #     print(X)
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

if __name__ == "__main__":
    unittest.main()


