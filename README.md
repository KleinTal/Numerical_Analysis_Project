
# Numerical Analysis and Optimization - Data Engineering and Science Bachelor's Program

This repository contains my solutions for the Numerical Analysis assignments, which were part of the Data Engineering and Science Bachelor's Program. The final project involved a competition among students to showcase their skills in Numerical Analysis and Optimization.

## Achievements

🏆 Secured the first place in the Numerical Analysis and Optimization competition.


## Table of Contents

- [Assignments Overview](#assignments-overview)
  - [Assignment 1](#assignment-1-14pt-interpolation)
  - [Assignment 2](#assignment-2-14pt-function-intersections)
  - [Assignment 3](#assignment-3-36pt-integration-and-area-calculation)
  - [Assignment 4](#assignment-4-14pt-least-squares-fitting)
  - [Assignment 5](#assignment-5-27pt-shape-area-calculation)
- [Test Functions](#test-functions)


## Assignments Overview

### Assignment 1: Interpolation

- Implement `Assignment1.interpolate(f, a, b, n)`.
- Return an interpolated function `g` based on the provided function `f`, range `[a, b]`, and number of points `n`.
- Testing will evaluate the interpolation errors.

### Assignment 2: Function Intersections

- Implement `Assignment2.intersections(f1, f2, maxerr)`.
- Return an iterable of approximate intersection points `Xs` between two functions `f1` and `f2` with error less than `maxerr`.

### Assignment 3: Integration and Area Calculation

- Implement `Assignment3.integrate(f, a, b, n)` to approximate the integral of a function `f` over the range `[a, b]`.
- Implement `Assignment3.areabetween(f1, f2)` to calculate the area between two functions `f1` and `f2`.

### Assignment 4: Least Squares Fitting

- Implement `Assignment4.fit(f, a, b, d, maxtime)` for least squares fitting of noisy data sampled from a function `f`.
- Return a function `g` that matches the clean version of the given function. 
- Use polynomial fitting with specified range `[a, b]` and expected degree `d`.
- Maximize efficiency within the provided `maxtime`.

### Assignment 5: Shape Area Calculation

- Implement `Assignment5.area(contour, maxerr)` to approximate the area of a shape.
- The shape contour can be sampled by calling with the desired number of points.
- Optimize for both accuracy and running time.

## Test Functions

- The assignments will be tested using various combinations of test functions.
- Test cases include polynomials and specific functions `f_1` to `f_11` defined in the assignment instructions.

- All assignments was graded based on accuracy of numerical solutions and running time.

