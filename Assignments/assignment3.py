"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and
the leftmost intersection points of the two functions.

The functions for the numeric answers are specified in MOODLE.


This assignment is more complicated than Assignment1 and Assignment2 because:
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors.
    2. You have the freedom to choose how to calculate the area between the two functions.
    3. The functions may intersect multiple times. Here is an example:
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately.
       You should explain why in one of the theoretical questions in MOODLE.

"""
import math

import numpy as np
import time
import random
import assignment2
import assignment1


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
         """
        # calculate the definite integral of f(x) using the Gaussian quadrature
        def LegendrePolynomial(x, n):
            if n == 0:
                return 1.0
            elif n == 1:
                return x
            else:
                final_poly = 1  # final Legendre polynomial
                last_poly = 0  # last Legendre polynomial
                for j in range(n):
                    helper_poly = last_poly
                    last_poly = final_poly
                    final_poly = ((2 * j + 1) * x * last_poly - j * helper_poly) / (j + 1)
                derive_poly = n * (x * final_poly - last_poly) / (x * x - 1)
                return final_poly, derive_poly

        def LegendreRoots(n):
            roots = [0.0 for j in range(n)]  # list with the roots
            for i in range(int((n+1)/2)):
                root = np.cos(np.pi * (i + 0.75) / (n + 0.5))  # initial guess
                while True:
                    fx = LegendrePolynomial(root, n)[0]
                    dfx = LegendrePolynomial(root, n)[1]
                    last_root = root
                    root = last_root - fx / dfx  # newton method for find roots
                    if abs(root - last_root) <= 1e-14:  # check if the error id goof enough
                        break
                roots[i] = -root  # root in the first iteration
                roots[n - 1 - i] = root
            return roots

        def LegendreWeights(roots, n):
            weights = [0.0 for i in range(n)]  # list with the weights
            for i in range(int((n+1)/2)):
                weights[i] = 2 / ((1 - roots[i] ** 2) * (LegendrePolynomial(roots[i], n)[1]) ** 2)
                weights[n - 1 - i] = weights[i]
            return weights

        roots = LegendreRoots(n)
        weights = LegendreWeights(roots, n)
        # roots, weights = np.polynomial.legendre.leggauss(n)
        for i in range(n):
            roots[i] = 0.5 * (b - a) * roots[i] + 0.5 * (b + a)  # scale the roots from the interval [-1, 1] to [a, b]
            weights[i] = 0.5 * (b - a) * weights[i]  # scale the weights from the interval [-1, 1] to [a, b]

        # Calculate the integral by summing up the multiply of weights and function values
        result = 0
        for i in range(n):
            result = result + weights[i] * f(roots[i])
        return np.float32(result)



    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        def intersection(f1, f2, a, b, maxerr=0.0001):

            # Implementation of Newton's method and Bisection method to find intersection points

            # the function for find the roots
            function = lambda x: f1(x) - f2(x)

            def derivative(x):
                h = 1e-10
                return (function(x + h) - function(x)) / h

            result = []
            rang = np.linspace(a, b, 300)
            # check points in the range
            for x in range(len(rang) - 1):
                n = 0
                x_i = rang[x]  # for newton
                a_i = rang[x]  # for bisection
                b_i = rang[x + 1]  # for bisection
                flag = 0  # if 1, the newton doesnt work and go to bisection
                last = x_i
                # just if this happened - there is root between those points
                if function(rang[x]) * function(rang[x + 1]) <= 0.0000001:
                    # try newton
                    while abs(function(x_i)) >= maxerr:
                        z = derivative(x_i)
                        if z == 0.0 or math.isnan(z):
                            flag = 1
                            break
                        x_i = x_i - function(x_i) / z
                        if last == x_i:  # python could not support this root
                            flag = 1
                            break
                        if n == 100:  # too many iteration, newton fails
                            flag = 1
                            break
                        n = n + 1
                        last = x_i
                    # newton doesnt work, try bisection
                    if flag == 1:
                        x = (a_i + b_i) / 2
                        while abs(b_i - a_i) / 2 > maxerr:
                            if function(x) == 0:
                                result.append(x)
                            if function(a) is None or function(x) is None:
                                break
                            if abs(function(x)) < maxerr:
                                result.append(x)
                                break
                            if function(a_i) * function(x) < 0:
                                b_i = x
                            else:
                                a_i = x
                            x = (a_i + b_i) / 2
                        x = (a_i + b_i) / 2
                        if abs(function(x)) < maxerr:
                            result.append(x)
                    # newton work, add the point
                    else:
                        result.append(x_i)
            return result

        def integral(f, a, b, n):
            # Implementation of numerical integration using the trapezoidal rule

            if a <= b:
                h = (b - a) / n
            else:
                h = (a - b) / n

            f_0 = 0
            f_1 = 0
            # f_2 = f(b)
            for i in range(1, n):
                x = a + i * h
                f_x = f(x)
                if i % 2 == 0:
                    # f_2 = f_2 + f_x
                    f_0 = f_0 + f_x
                else:
                    f_1 = f_1 + f_x

            result = (h / 3) * (2 * f_0 + 4 * f_1 + f(a) + f(b))
            return np.float32(result)

        fun1 = lambda x: f1(x) - f2(x)
        fun2 = lambda x: f2(x) - f1(x)

        inter = intersection(f1, f2, 1, 100)  # find intersection

        if len(inter) < 2 or len(inter) > 10000:  # no area
            return np.float32(None)

        inter = sorted(inter)
        area = 0
        for i in range(len(inter) - 1):

            if f1(i + 0.1) > f2(i + 0.1):  # f1 > f2

                area = area + integral(fun1, inter[i], inter[i + 1], 500)

            else:  # f2 > f1
                area = area + integral(fun2, inter[i], inter[i + 1], 500)
        area = np.float32(abs(area))
        return area


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f = np.poly1d([-1, 0, 1])
        f = lambda x: 1 / log(x)
        r = ass3.integrate(f, 1.5, 12, 10)
        a = abs(r - 6.875482635096272)
        print(a)

    def test_integrate_float321(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f2 = lambda x: math.sqrt((1 - x ** 4))
        r = ass3.integrate(f2, 0, 1, 12)
        print(r, "f3")

    def test_integrate_float322(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: log(6 * log(x))
        r = ass3.integrate(f, 2, 5, 10)
        print(r, f"4")

    def test_integrate_float323(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: e ** (-x) / x
        r = ass3.integrate(f, 2, 5, 4)
        print(r, "f5")

    def test_integrate_float324(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: x ** (e - 1) * e ** (-x)
        r = ass3.integrate(f, 2, 5, 4)
        print(r, "f6")

    def test_integrate_float325(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: np.sin(x) * math.sqrt(3 - x ** 2 + x ** 3)
        r = ass3.integrate(f, 0, 2 * math.pi, 5)
        print(r, "f7")

    def test_integrate_float326(self):
        ass3 = Assignment3()
        # f1 = np.poly1d([-1, 0, 1])
        f1 = lambda x: pow(x, 2) - 3 * x + 2
        f2 = lambda x: np.sin(log(x))
        r = ass3.areabetween(f1, f2)
        print(r, "f8")

    def test_integrate_float327(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: pow(e, -2 * pow(x, 2))
        r = ass3.integrate(f, -2, 2, 8)
        print(r, "1.253234")

    def test_integrate_float328(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: np.arctan(x)
        r = ass3.integrate(f, -5, 10, 10)
        print(r, "7.1657609")

    def test_integrate_float329(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: np.sin(x) / x
        r = ass3.integrate(f, -7, 3.14, 120)
        print(r, "0.1649406")

    def test_integrate_float3210(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: 1 / log(x)
        r = ass3.integrate(f, 1.2, 20, 11)
        print(r, "10.839087")

    def test_integrate_float3211(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: pow(e, pow(e, x))
        r = ass3.integrate(f, -10, 2, 11)
        print(r, "273.3102885")

    def test_integrate_float3212(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: np.sin(log(x))
        r = ass3.integrate(f, 0.1, 60, 20)
        print(r, "10.650683")

    def test_integrate_float3213(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = lambda x: log(log(x))
        r = ass3.integrate(f, 2, 60, 10)
        print(r, "65.3891764")

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
