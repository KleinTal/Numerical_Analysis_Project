"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""

import numpy as np
import random
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape

    def __init__(self, area):
        self.my_area = area

    def sample(self):
        pass

    def contour(self, n: int):
        return

    def area(self):
        return np.float32(self.my_area)


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        def area_calc(num):

            # Function to calculate the area of the shape using a contour
            area = contour(num)
            sum_area = 0
            for j in range(len(area)):
                x1 = area[j - 1][0]
                y1 = area[j - 1][1]
                x2 = area[j][0]
                y2 = area[j][1]
                if x1 != x2:
                    sum_area += (x2 - x1) * (y1 + y2)
            return sum_area / 2

        n = 100
        # Starting with an initial number of points (n)
        my_area = area_calc(n)
        # Calculate the area using the current number of points

        while True:
            # Iterate until a suitable level of accuracy is achieved
            n += 100
            # Increase the number of points for the next iteration
            real_area = area_calc(n)
            # Calculate the area with the increased number of points
            error = abs(abs(my_area) - abs(real_area)) / abs(real_area)
            # Calculate the relative error between consecutive area calculations

            if error < maxerr:
                # Check if the relative error is below the specified threshold
                break
            # Update the area value for the next iteration
            my_area = real_area

        # Assign the final calculated area
        my_area = real_area
        # Return the final area value as a 32-bit floating-point number
        return np.float32(my_area)

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        def euclidean(point, data):
            """
            Euclidean distance between point & data.
            Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
            """
            return np.sqrt(np.sum((point - data) ** 2, axis=1))

        def kmeans(X_train, n_clusters):
            # K-Means clustering algorithm to find centroids

            # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
            # then the rest are initialized w/ probabilities proportional to their distances to the first
            # Pick a random point from train data for first centroid
            centroids = [random.choice(X_train)]
            for _ in range(n_clusters - 1):
                # Calculate distances from points to the centroids
                dists = np.sum([euclidean(centroid, X_train) for centroid in centroids], axis=0)
                # Normalize the distances
                dists /= np.sum(dists)
                # Choose remaining points based on their distances
                new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
                centroids += [X_train[new_centroid_idx]]
            # Iterate, adjusting centroids until converged or until passed max_iter
            iteration = 0
            prev_centroids = None
            while np.not_equal(centroids, prev_centroids).any() and iteration < 300:
                # Sort each datapoint, assigning to nearest centroid
                sorted_points = [[] for _ in range(n_clusters)]
                for x in X_train:
                    dists = euclidean(x, centroids)
                    centroid_idx = np.argmin(dists)
                    sorted_points[centroid_idx].append(x)
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                prev_centroids = centroids
                centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
                for i, centroid in enumerate(centroids):
                    if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                        centroids[i] = prev_centroids[i]
                iteration += 1
            return centroids

        def counterclockwise_sort(list_of_xy_coords):

            # Sort coordinates in a counter-clockwise order
            cx, cy = list_of_xy_coords.mean(0)
            x, y = list_of_xy_coords.T
            angles = np.arctan2(x - cx, y - cy)
            indices = np.argsort(angles)
            return list_of_xy_coords[indices]

        # Sampling points
        points = []
        for i in range(2000):
            point = sample()
            points.append(point
                          )
        # K-Means clustering to find centroids
        points = np.array(points)
        centers = np.array(kmeans(points, 30))
        points = counterclockwise_sort(centers)

        def area_calc(area):
            # Function to calculate the area of the shape using centroids
            sm = 0
            # Initialize a variable to accumulate the signed area

            for j in range(len(area)):
                # Iterate over the vertices of the shape
                x1 = area[j - 1][0]
                y1 = area[j - 1][1]
                x2 = area[j][0]
                y2 = area[j][1]
                # Retrieve coordinates of two consecutive vertices

                if x1 != x2:
                    # Ensure the vertices are distinct along the x-axis to avoid division by zero
                    sm += (x2 - x1) * (y1 + y2)
                    # Accumulate the signed area using the trapezoidal rule

            return sm / 2
            # Return the final calculated area by dividing the accumulated signed area by 2

        area = area_calc(points)
        # Calculate the signed area of the shape using the provided function
        result = MyShape(area)
        # Create an instance of MyShape using the calculated area
        return result
        # Return the created shape instance


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(abs(a) - abs(np.pi)) / abs(np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(abs(a) - abs(np.pi)) / abs(np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
