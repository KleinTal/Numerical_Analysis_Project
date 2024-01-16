"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""
import math

import numpy as np
import time


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        # Time measurement for breaking the loop if it takes too long

        time0 = time.time()
        exam = np.linspace(a, b, 1000000)
        i = 0
        for ex in exam:
            f(ex)
            time1 = time.time()
            tm = time1 - time0
            if tm >= 0.5:
                break
            i += 1

        # Determine the number of points based on elapsed time
        if i == 0:
            n = math.floor((maxtime - tm) / tm)
        else:
            n = i * math.floor(maxtime / 0.6)

        # Cap the number of points to avoid exceeding time limits
        if n >= 6000000:
            n = 6000000
        elif n >= 4000000:
            n = 4000000
        elif n >= 2000000:
            n = 2000000

        # Generate x values and corresponding y values
        x_values = np.linspace(a, b, n)
        y_values = []
        for x in x_values:
            y_values.append(f(x))
        y_values = np.array(y_values)

        # Construct the Vandermonde matrix and solve for coefficients
        vander = np.vander(x_values, d + 1)
        ans = np.linalg.inv(np.transpose(vander).dot(vander)).dot(np.transpose(vander)).dot(y_values)

        # Define the result function using the computed coefficients
        def result(x):
            x_s = np.transpose(np.vander([x], d + 1))
            return np.dot(ans,x_s)[0]
        return result

        ##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from commons import *
import threading
import _thread as thread


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = lambda x: log(log(x))
        nf = NOISY(10)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=5, b=10, d=10, maxtime=20)
        T = time.time() - T
        mse = 0
        for x in np.linspace(5, 10, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print('mas: ', mse)

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        print(T)
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(0.88)(NOISY(0.01)(poly(1, 1, 1)))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=10)
        T = time.time() - T
        self.assertLessEqual(T, 10)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
