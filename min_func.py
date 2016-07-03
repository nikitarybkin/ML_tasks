__author__ = 'Nikita Rybkin'

import math
import numpy
import random

lambda_c = 0.00003
lambda_delta = 0.99999
EPS = 1e-5
minX = -10
maxX = 10
MAXIterations = 1e5


def lambda_counter(x, grad, function):
	l = 0
	r = 1
	while (r - l > EPS):
		x1 = l + (r - l) / 3
		x2 = l + (r - l) / 3 * 2
		f1 = function(substract(x, vector_multiplicator(grad, x1)))
		f2 = function(substract(x, vector_multiplicator(grad, x2)))
		if (f1 > f2):
			l = x1
		else:
			r = x2
	f1 = function(substract(x, vector_multiplicator(grad, l)))
	f2 = function(substract(x, vector_multiplicator(grad, r)))
	if (f1 < f2):
		return l
	else:
		return r


def rozen_two(x):
	return ((1 - x[0]) ** 2) + 100 * ((x[1] - x[0] ** 2) ** 2)


def rozen_multi(n):
	def f(x):
		res = 0
		for i in range(int(n / 2)):
			res = res + (1 - x[2 * i]) ** 2 + 100 * ((x[2 * i + 1] - x[2 * i] ** 2) ** 2)
		return res

	return f


def rozen_grad_two(x):
	res = []
	res.append(2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1))
	res.append(200 * (x[1] - x[0] ** 2))
	return res


def rozen_grad_multi(n):
	def grad(x):
		res = []
		for i in range(int(n / 2)):
			res.append(2 * (200 * x[2 * i] ** 3 - 200 * x[2 * i] * x[2 * i + 1] + x[2 * i] - 1))
			res.append(200 * (x[2 * i + 1] - x[2 * i] ** 2))
		return res

	return grad


def substract(x1, x2):
	x = []
	for i in range(len(x1)):
		x.append(x1[i] - x2[i])
	return x


def vector_multiplicator(x, k):
	res = []
	for i in x:
		res.append(i * k)
	return res


def grad_desc(x, function, grad_func, method):
	Lambda = lambda_c
	if (method == 3):
		Lambda = lambda_counter(x, grad_func(x), function)
	nextX = substract(x, vector_multiplicator(grad_func(x), Lambda))
	Iterations = 0
	while (numpy.linalg.norm(substract(x, nextX)) > EPS
		   and math.fabs(function(x) - function(nextX)) > EPS
		   and numpy.linalg.norm(grad_func(x)) > EPS
		   and Iterations < MAXIterations):
		x = nextX
		if (method == 2):
			Lambda = Lambda * lambda_delta
		if (method == 3):
			Lambda = lambda_counter(x, grad_func(x), function)
		nextX = substract(x, vector_multiplicator(grad_func(x), Lambda))
		if (Iterations > 100000):
			print(Iterations)
		Iterations += 1
	x = nextX
	return x


def monte_carlo(n, v_length, function, grad_func, method):
	minF = math.inf
	for i in range(n):
		x = []
		for j in range(v_length):
			x.append(random.uniform(minX, maxX))
		print("x0 == ", x)
		x = grad_desc(x, function, grad_func, method)
		print("x == ", x, "\tf(x) == ", function(x), "\n------------------------")
		if function(x) < minF:
			minF = function(x)
			minArg = x
	return minArg

if __name__ == '__main__':
    print("Two-dimensional Rozenbrock")
    minArg = monte_carlo(5, 2, rozen_two, rozen_grad_two, 1)
    print("x = ", minArg)
    print("f(x) = ", rozen_two(minArg))

    print("Multidimensional Rozenbrock")
    minArg = monte_carlo(5, 2, rozen_multi(2), rozen_grad_multi(2), 1)
    print("x = ", minArg)
    print("f(x) = ", rozen_multi(2)(minArg))
