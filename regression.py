__author__ = 'Nikita Rybkin'

import numpy
import math
import cPickle as pickle
import pylab


def corr(data1, data2):
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()
    corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr

def linear_regression(x, y):
    m = x.shape[0]
    matrix = numpy.ndarray(shape=(m, 2))
    for j in range(m):
        matrix[j, 0] = 1
        matrix[j, 1] = x[j].tolist()[0]
    b = beta_counter(matrix, y)
    def f(t):
        p = b[0] + b[1] * t
        return p
    return f


def polynomial_regression(x, y, k):
    m = x.shape[0]
    matrix = numpy.ndarray(shape=(m, k + 1))
    for j in range(m):
        matrix[j, 0] = 1
        for z in range(1, k + 1):
            matrix[j, z] = x[j].tolist()[0] ** z
    b = beta_counter(matrix, y)
    def f(t):
        p = 0
        for i in range(k + 1):
            p += b[i] * (t ** i)
        return p
    return f

def vector_operator(f, v):
    z = numpy.ndarray(v.shape)
    for i in range(len(v)):
        z[i] = f(v[i])
    return z

def functional_regression(x, y, functions):
    m = x.shape[0]
    functions = numpy.concatenate(([id], functions))
    k = len(functions)
    matrix = numpy.ndarray(shape=(m, k))
    for j in range(m):
        for z in range(k):
            matrix[j, z] = functions[z](x[j].tolist()[0])
    b = beta_counter(matrix, y)
    def f(t):
        p = 0
        for i in range(k):

            p += b[i] * vector_operator(functions[i], t)
        return p
    return f


def beta_counter(x, y):
    b = numpy.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return b

def sqr(x):
    return x**2

def eq(x):
    return x

def id(x):
    return 1

with open('task2_dataset_3.txt', 'rb') as f:
    x, y = pickle.load(f)
pylab.plot(x, y, 'go')
pylab.show()
print (corr(linear_regression(x,y)(x), y))
print (corr(polynomial_regression(x,y,2)(x), y))
print (corr(functional_regression(x,y,[math.sin, math.cos, math.tan])(x), y))


'''def leave_one_out_cv (regression_type, x, y):
    learn_x = numpy.ndarray(len(x)-1)
    learn_y = numpy.ndarray(len(y)-1)
    check_x = 0
    check_y = 0
    q = 0
    for i in range(len(x)):
        check_x = x[i]
        check_y = y[i]
        for j in range (i):
            learn_x[j] = x[j]
            learn_y[j] = y[j]
        for j in range(i+1, len(learn_x)):
            learn_x[j] = x[j]
            learn_y[j] = y[j]
        if(regression_type == 'linear'):
            q += corr(linear_regression(check_x, check_y)(check_x), check_y)
        if (regression_type == 'polynomial'):
            q += corr(polynomial_regression(check_x, check_y, 2)(check_x), check_y)
        if (regression_type == 'functional'):
            q += corr(functional_regression(check_x, check_y, [math.sin, math.cos])(check_x), check_y)
    q /= len(x)
    print q

leave_one_out_cv ('linear', x, y)'''