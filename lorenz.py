from scipy.linalg import norm
import numpy as np
import pickle
import scipy.io as io
import os, inspect
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from uq import stationary_statistical_learning_reduced

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def objfun(s,r,b,stepCnt = 1000, dt = 0.01):
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))
    # Setting initial values
    # xs[0], ys[0], zs[0] = (0., 1., 1.05)
    xs[0], ys[0], zs[0] = np.random.rand(3)
    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    # mu =np.mean(zs)
    # mu = np.vstack((mu,np.mean(zs)))
    # return np.mean(zs)+np.mean(zs**2), np.mean(zs**2), np.mean(zs)


    return np.mean(zs) + np.std(zs), np.std(zs), np.mean(zs)




def objfun0(s,r,b,stepCnt = 1000, dt = 0.01):
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))
    # Setting initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    # xs[0], ys[0], zs[0] = np.random.rand(3)
    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    # mu =np.mean(zs)
    # mu = np.vstack((mu,np.mean(zs)))
    # return np.mean(zs)-25.2696 #+np.mean(zs**2)/694.1202
    return np.mean(zs) + np.std(zs)  # +np.mean(zs**2)/694.1202


def RK4_timemarch(x, fun, h):
    '''
    This is an implementation of the fourth-order Runge-Kutta method
to solve systems of ODEs.
    :param x: the initial point
    :param fun: objective function
    :param h: step size
    :return: the time marched point
    '''
    f1=fun(x)
    f2=fun(x+h/2*f1)
    f3=fun(x+h/2*f2)
    f4=fun(x+h*f3)
    x1=x+h/6*(f1+2*f2+2*f3+f4)
    return x1

def lorenz2(x, s=10, r=28, b=2.667):
    x_dot = np.empty((3,))
    x_dot[0] = s*(x[1] - x[0])
    x_dot[1] = r*x[0] - x[1] - x[0]*x[2]
    x_dot[2] = x[0]*x[1] - b*x[2]
    return x_dot


def lorenz_lost2(xi, T, h, bnd2, bnd1, y0 = 0,  method=1, ind_exist=-1):
    '''

    :param x0: (s,r,b) point of interest. This is normalized 0<x0<1
    :param T:  total attractor time
    :param h: step size
    :param ind_exist: index of point from evaluated set
    :param y0: target value
    :return:
    '''
    s0 = 10
    r0 = 28
    b0 = 2.667
    xiT = xi.reshape(-1, 1)
    n = xiT.shape[0]
    x0 = np.copy(xi)
    # for ii in range(n):
    #     x0[ii] = xi[ii]*(bnd2[ii] - bnd1[ii]) + bnd1[ii]

    if n == 1:
        x = np.hstack((s0, x0, b0))
    elif n == 2:
        x = np.hstack((s0, x0))
    else:
        x = x0

    stepCnt = int(T/h*1.0)

    if ind_exist == -1:
        # start a new simulation
        xs = np.zeros(stepCnt)
        ys = np.zeros(stepCnt)
        zs = np.zeros(stepCnt)
        # Setting initial values
        xs[0] = 0
        ys[0] = 1
        zs[0] = 1.05
    else:
        xs = np.zeros(stepCnt)
        ys = np.zeros(stepCnt)
        zs = np.zeros(stepCnt)

        pt = io.loadmat('allpoints/pt_to_eval' + str(ind_exist) + '.mat')
        xs0 = pt['xs'][0]
        xs[0] = xs0[-1]
        ys0 = pt['ys'][0]
        ys[0] = ys0[-1]
        zs0 = pt['zs'][0]
        zs[0] = zs0[-1]
        # xs[0], ys[0], zs[0] = np.random.rand(3)
        # Stepping through "time".
    if method == 0:  # FW Euler
        for i in range(stepCnt - 1):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], x[0], x[1], x[2])
            xs[i + 1] = xs[i] + (x_dot * h)
            ys[i + 1] = ys[i] + (y_dot * h)
            zs[i + 1] = zs[i] + (z_dot * h)
    elif method == 1:  # RK4
        fun = lambda X: lorenz2(X, x[0], x[1], x[2])
        for i in range(stepCnt - 1):
    # % Derivatives of the X, Y, Z state
            X = np.hstack((xs[i], ys[i], zs[i]))
            x1 = RK4_timemarch(X, fun, h)
            xs[i+1] = x1[0]
            ys[i+1] = x1[1]
            zs[i+1] = x1[2]
    # existing point add this simulation to it
    # ind_exist = # of points + 1

    if ind_exist > 0:
        xs = np.hstack((xs0, xs[1:]))
        ys = np.hstack((ys0, ys[1:]))
        zs = np.hstack((zs0, zs[1:]))

        # pt = {'zs': zs, 'ys': ys, 'xs': xs, 'index': ind_exist, 'h': h, 'T': T}
        # io.savemat('allpoints/pt_to_eval' + str(ind_exist) + '.mat', pt)
    # else:

        # var_opt = io.loadmat("allpoints/pre_opt")
        # ind_exist = var_opt['num_point'][0, 0]
        # pt = {'zs': zs, 'ys': ys, 'xs': xs, 'index': ind_exist, 'h': h, 'T': T}
        # io.savemat('allpoints/pt_to_eval' + str(ind_exist) + '.mat', pt)

    if n == 1:
        J = np.abs(np.mean(zs)-y0)[0]
        return J, zs, ys, xs
    elif n == 2:
        J = (np.sum(((np.mean(zs)-y0[0])**2, (np.std(zs)-y0[0])**2)))/len(y0)*1.0
        return J, zs, ys, xs
    else:
        J = (np.sum(((np.mean(zs) - y0[0]) ** 2, (np.std(zs) - y0[0]) ** 2))) / len(y0) * 1.0
        return J, zs, ys, xs

