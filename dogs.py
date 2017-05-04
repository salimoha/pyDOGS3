import csv
import numpy as np
import scipy
import pandas as pd
from scipy import optimize
from scipy.spatial import Delaunay
import scipy.io as io
from configobj import ConfigObj
import configparser
import pickle
import os, inspect
import uq
from tr import transient_removal
import lorenz
import tr

np.set_printoptions(linewidth=200, precision=5, suppress=True)
pd.options.display.max_rows = 20
pd.options.display.expand_frame_repr = False

'''MIT License
Copyright (c) 2017
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Authors: Shahrouz Alimo, Muhan Zhao
Modified: Feb. 2017 '''


class pointsdata:
    index = 0
    cost_comp = 0
    simTime = 0
    N = 0
    xE = 0
    yE = 0
    sigma = 0
    yreal = 0


def bounds(bnd1, bnd2, n):
    #   find vertex of domain for a box domain.
    #   INPUT: n: dimension, bnd1: lower bound, bnd2: upper bound.
    #   OUTPUT: vertex of domain. 2^n number vector of n-D.
    #   Example:
    #           n = 3
    #           bnd1 = np.zeros((n, 1))
    #           bnd2 = np.ones((n, 1))
    #           bnds = bounds(bnd1,bnd2,n)
    #   Author: Shahoruz Alimohammadi
    #   Modified: Dec., 2016
    #   DELTADOGS package
    bnds = np.kron(np.ones((1, 2 ** n)), bnd2)
    for ii in range(n):
        tt = np.mod(np.arange(2 ** n) + 1, 2 ** (n - ii)) <= 2 ** (n - ii - 1) - 1
        bnds[ii, tt] = bnd1[ii];
    return bnds


def mindis(x, xi):
    '''
    calculates the minimum distance from all the existing points
    :param x: x the new point
    :param xi: xi all the previous points
    :return: [ymin ,xmin ,index]
    '''
    #
    # %
    # %
    # %
    x = x.reshape(-1, 1)
    y = float('inf')
    index = float('inf')
    x1 = np.copy(x) * float('inf')
    N = xi.shape[1]
    for i in range(N):
        y1 = np.linalg.norm(x[:, 0] - xi[:, i])
        if y1 < y:
            y = np.copy(y1)
            x1 = np.copy(xi[:, i])
            index = np.copy(i)
    return y, index, x1


def modichol(A, alpha, beta):
    #   Modified Cholesky decomposition code for making the Hessian matrix PSD.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    n = A.shape[1]  # size of A
    L = np.identity(n)
    ####################
    D = np.zeros((n, 1))
    c = np.zeros((n, n))
    ######################
    D[0] = np.max(np.abs(A[0, 0]), alpha)
    c[:, 0] = A[:, 0]
    L[1:n, 0] = c[1:n, 0] / D[0]

    for j in range(1, n - 1):
        c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
        for i in range(j + 1, n):
            c[i, j] = A[i, j] - (np.dot((L[i, 0:j] * L[j, 0:j]).reshape(1, j), D[0:j]))[0, 0]
        theta = np.max(c[j + 1:n, j])
        D[j] = np.array([(theta / beta) ** 2, np.abs(c[j, j]), alpha]).max()
        L[j + 1:n, j] = c[j + 1:n, j] / D[j]
    j = n - 1;
    c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
    D[j] = np.max(np.abs(c[j, j]), alpha)
    return np.dot(np.dot(L, (np.diag(np.transpose(D)[0]))), L.T)


def circhyp(x, N):
    # circhyp     Circumhypersphere of simplex
    #   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
    #   and the square of the radius of the N-dimensional hypersphere
    #   encircling the simplex defined by its N+1 vertices.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    #   DOGS package

    test = np.sum(np.transpose(x) ** 2, axis=1)
    test = test[:, np.newaxis]
    m1 = np.concatenate((np.matrix((x.T ** 2).sum(axis=1)), x))
    M = np.concatenate((np.transpose(m1), np.matrix(np.ones((N + 1, 1)))), axis=1)
    a = np.linalg.det(M[:, 1:N + 2])
    c = (-1.0) ** (N + 1) * np.linalg.det(M[:, 0:N + 1])
    D = np.zeros((N, 1))
    for ii in range(N):
        M_tmp = np.copy(M)
        M_tmp = np.delete(M_tmp, ii + 1, 1)
        D[ii] = ((-1.0) ** (ii + 1)) * np.linalg.det(M_tmp)
        # print(np.linalg.det(M_tmp))
    # print(D)
    xC = -D / (2.0 * a)
    #	print(xC)
    R2 = (np.sum(D ** 2, axis=0) - 4 * a * c) / (4.0 * a ** 2)
    #	print(R2)
    return R2, xC


########################################SURROGATE MODELS#################################################

class Inter_par():
    def __init__(self, method="NPS", w=0, v=0, xi=0, a=0):
        self.method = "NPS"
        self.w = []
        self.v = []
        self.xi = []
        self.a = []


def interpolateparameterization(xi, yi, inter_par):
    n = xi.shape[0]
    m = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros(shape=(m, m))
        for ii in range(0, m, 1):  # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(0, m, 1):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)

        V = np.concatenate((np.ones((1, m)), xi), axis=0)
        A1 = np.concatenate((A, np.transpose(V)), axis=1)
        A2 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
        yi = yi[np.newaxis, :]
        # print(yi.shape)
        b = np.concatenate([np.transpose(yi), np.zeros(shape=(n + 1, 1))])
        #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
        A = np.concatenate((A1, A2), axis=0)
        wv = np.linalg.solve(A, b)
        inter_par.w = wv[:m]
        inter_par.v = wv[m:]
        inter_par.xi = xi
        return inter_par



def regressionparametarization(xi, yi, sigma, inter_par):
    # Notice xi, yi and sigma must be a two dimension matrix, even if you want it to be a vector.
    # or there will be error
    n = xi.shape[0]
    N = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros(shape=(N, N))
        for ii in range(N):  # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(N):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)
        V = np.concatenate((np.ones((1, N)), xi), axis=0)
        w1 = np.linalg.lstsq(np.dot(np.diag(1 / sigma), V.T), (yi / sigma).reshape(-1, 1))
        w1 = np.copy(w1[0])
        b = np.mean(np.divide(np.dot(V.T, w1) - yi.reshape(-1, 1), sigma.reshape(-1, 1)) ** 2)
        wv = np.zeros([N + n + 1])
        if b < 1:
            wv[N:] = np.copy(w1.T)
            rho = 1000
            wv = np.copy(wv.reshape(-1, 1))
        else:
            rho = 1.1
            fun = lambda rho: smoothing_polyharmonic(rho, A, V, sigma, yi, n, N, 1)
            rho = optimize.fsolve(fun, rho)
            b, db, wv = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N, 3)
        inter_par.w = wv[:N]
        inter_par.v = wv[N:N + n + 1]
        inter_par.xi = xi
        yp = np.zeros(N)
        while (1):
            for ii in range(N):
                yp[ii] = interpolate_val(xi[:, ii], inter_par)
            residual = np.max(np.divide(np.abs(yp - yi), sigma))
            if residual < 2:
                break
            rho *= 0.9
            b, db, wv = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N, 3)
            inter_par.w = wv[:N]
            inter_par.v = wv[N:N + n + 1]
    return inter_par, yp


def smoothing_polyharmonic(rho, A, V, sigma, yi, n, N, num_arg):
    # Notice: num_arg = 1 will return b
    #         num_arg = 2 will return db
    #         num_arg = else will return b,db,wv
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1, 1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.solve(A1, b1)
    b = np.mean(np.multiply(wv[:N], sigma.reshape(-1, 1)) ** 2 * rho ** 2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N], sigma.reshape(-1, 1) ** 2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.solve(-A1, bdwv)
    db = 2 * np.mean(np.multiply(wv[:N] ** 2 * rho + rho ** 2 * np.multiply(wv[:N], Dwv[:N]), sigma ** 2))
    if num_arg == 1:
        return b
    elif num_arg == 2:
        return db
    else:
        return b, db, wv


def interpolate_hessian(x, inter_par):
    if inter_par.method == "NPS" or self.method == 1:
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        n = x.shape[0]

        H = np.zeros((n, n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            if np.linalg.norm(X) > 1e-5:
                H = H + 3 * w[ii] * ((X * X.T) / np.linalg.norm(X) + np.linalg.norm(X) * np.identity(n))
        return H


def interpolate_val(x, inter_par):
    # Each time after optimization, the result value x that optimization returns is one dimension vector,
    # but in our interpolate_val function, we need it to be a two dimension matrix.
    x = x.reshape(-1, 1)
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        x1 = np.copy(x)
        x = pd.DataFrame(x1).values
        S = xi - x
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (
            np.sqrt(np.diag(np.dot(S.T, S))) ** 3))


def interpolate_grad(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros([n, 1])
        x1 = np.copy(x)
        x = pd.DataFrame(x1).values
        for ii in range(N):
            X = x - xi[:, ii].reshape(-1, 1)
            g = g + 3 * w[ii] * X * np.linalg.norm(X)
        g = g + v[1:]

    return g


def inter_min(x, inter_par, Ain=[], bin=[]):
    # %find the minimizer of the interpolating function starting with x
    rho = 0.9  # backtracking paramtere
    n = x.shape[0]
    #     start the serafh method
    iter = 0
    x0 = np.zeros((n, 1))
    # while iter < 10:
    H = np.zeros((n, n))
    g = np.zeros((n, 1))
    y = interpolate_val(x, inter_par)
    g = interpolate_grad(x, inter_par)
    # H = interpolate_hessian(x, inter_par)
    # Perform the Hessian modification
    # H = modichol(H, 0.01, 20);
    # H = (H + H.T)/2.0
    #         optimizaiton for finding hte right direction
    objfun3 = lambda x: (interpolate_val(x, inter_par))
    grad_objfun3 = lambda x: interpolate_grad(x, inter_par)
    res = minimize(objfun3, x0, method='L-BFGS-B', jac=grad_objfun3, options={'gtol': 1e-6, 'disp': True})
    return res.x, res.fun


#################################### Constant K method ####################################


def tringulation_search_bound_constantK(inter_par, xi, K, ind_min):
    n = xi.shape[0]
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = scipy.spatial.Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]

    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        R2, xc = circhyp(xi[:, tri[ii, :]], n)
        x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
        Sc[ii] = interpolate_val(x, inter_par) - K * (R2 - np.linalg.norm(x - xc) ** 2)
        if np.sum(ind_min == tri[ii, :]):
            Scl[ii] = Sc[ii]
        else:
            Scl[ii] = np.inf
    # Global one
    t = np.min(Sc)
    ind = np.argmin(Sc)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
    xm, ym = Constant_K_Search(x, inter_par, xc, R2, K)
    # Local one
    t = np.min(Scl)
    ind = np.argmin(Scl)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    # Notice!! ind_min may have a problen as an index
    x = np.copy(xi[:, ind_min])
    xml, yml = Constant_K_Search(x, inter_par, xc, R2, K)
    if yml < ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
    return xm, ym


def Constant_K_Search(x0, inter_par, xc, R2, K, lb=[], ub=[]):
    n = x0.shape[0]
    costfun = lambda x: Contious_search_cost(x, inter_par, xc, R2, K, 1)
    costjac = lambda x: Contious_search_cost(x, inter_par, xc, R2, K, 2)
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    res = optimize.minimize(costfun, x0, jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


# value of consatn K search
def Contious_search_cost(x, inter_par, xc, R2, K, num_arg):
    # if num_arg == 1: return M
    # if num_arg == 2: return DM
    x = x.reshape(-1, 1)
    M = interpolate_val(x, inter_par) - K * (R2 - np.linalg.norm(x - xc) ** 2)
    DM = interpolate_grad(x, inter_par) + 2 * K * (x - xc)
    if num_arg == 1:
        return M
    if num_arg == 2:
        return DM.T


############################### Cartesian Lattice functions ######################

def ismember(A, B):
    return [np.sum(a == B) for a in A]


def points_neighbers_find(x, xE, xU, Bin, Ain):
    # delta_general, index1,x1 = mindis(x, np.concatenate((xE,xU ), axis=1) )
    x = x.reshape(-1, 1)
    x1 = mindis(x, np.concatenate((xE, xU), axis=1))[2].reshape(-1, 1)
    active_cons = []
    b = Bin - np.dot(Ain, x)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons.append(i + 1)
    active_cons = np.array(active_cons)

    active_cons1 = []
    b = Bin - np.dot(Ain, x1)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons1.append(i + 1)
    active_cons1 = np.array(active_cons1)

    if len(active_cons) == 0 or min(ismember(active_cons, active_cons1)) == 1:
        newadd = 1
        success = 1
        if mindis(x, xU)[0] == 0:
            newadd = 0
    else:
        success = 0
        newadd = 0
        xU = np.hstack((xU, x))
    return x, xE, xU, newadd, success


############################################ Test Examples ############################################
def read_str(S, judge):
    s = S[0]
    for i in range(len(s)):
        if s[i] != '=':
            continue
        elif s[i] == '=':
            i += 1
            break
    r = s[i:]
    if judge == 'i':
        r = int(r)
    else:
        r = float(r)
    return r


def solver(x, fun_arg, flag=1):
    # the noise level
    sigma0 = 0.3
    T0 = 1
    if fun_arg == 1:  # Quadratic
        funr = lambda x: 5 * np.linalg.norm(x - 0.3) ** 2
        fun = lambda x: 5 * np.linalg.norm(x - 0.3) ** 2 + sigma0 * np.random.randn()

    elif fun_arg == 2:  # Schwefel
        funr = lambda x: -sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250
        fun = lambda x: -sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250 + sigma0 * np.random.randn()
        if flag == 1: # new point

            y = fun(x)
            T = T0
            sigmaT = sigma0 / np.sqrt(T)
            return y, sigmaT, T

        else:

            fin = open("PtsToEval/surr_J_new.dat", "r")
            T_exist = int(fin.readline())
            y_exist = float(fin.readline())
            fin.close()

            T = T_exist + T0
            y = (fun(x) + y_exist * T_exist) / T
            # UQ method...!!!!!!
            sigmaT = sigma0 / np.sqrt(T)

            return y, sigmaT, T

        # rastinginn
    elif fun_arg == 3:
        A = 3
        funr = lambda x: sum((x - 0.7) ** 2 - A * np.cos(2 * np.pi * (x - 0.7)))
        fun = lambda x: sum((x - 0.7) ** 2 - A * np.cos(2 * np.pi * (x - 0.7))) + sigma0 * np.random.randn()

    elif fun_arg == 4:
        #     lorenz attractor
        var_opt = io.loadmat("allpoints/pre_opt")
        idx = var_opt['num_point'][0, 0]
        flag = var_opt['flag'][0, 0]
        bnd1 = var_opt['lb'][0]
        bnd2 = var_opt['ub'][0]
        T_lorenz = var_opt['T_lorenz'][0, 0]
        h = var_opt['h_lorenz'][0, 0]
        y0 = np.array([23.5712])
        time_method = 1
        DT = 10

        if flag == 1:   # flag 1 : new point

            J, zs, ys, xs = lorenz.lorenz_lost2(x, T_lorenz, h, bnd2, bnd1, y0, time_method)
            xx = uq.data_moving_average(zs, 40).values
            sigmaT = np.sqrt(uq.stationary_statistical_learning_reduced(xx, 18)[0])

            return J, zs, ys, xs, sigmaT, T_lorenz

        else:  # flag = 0: existing point

            data = io.loadmat("allpoints/pt_to_eval" + str(idx) + ".mat")
            T_zs_lorenz = data['T'][0]

            J, zs, ys, xs = lorenz.lorenz_lost2(x, T_zs_lorenz + DT, h, bnd2, bnd1, y0, time_method, idx)
            xx = uq.data_moving_average(zs, 40).values
            sigmaT = np.sqrt(uq.stationary_statistical_learning_reduced(xx, 18)[0])

            return J, zs, ys, xs, sigmaT, T_zs_lorenz + DT


#############################           LORENZ             ##################################################
def normalize_bounds(x0, lb, ub):
    n = len(lb)  # n represents dimensions
    m = x0.shape[1]  # m represents the number of sample data
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j] - lb[i]) / (ub[i] - lb[i])
    return x


def physical_bounds(x0, lb, ub):
    '''
    :param x0: normalized point
    :param lb: real lower bound
    :param ub: real upper bound
    :return: physical scale of the point
    '''
    n = len(lb)  # n represents dimensions
    try:
        m = x0.shape[1]  # m represents the number of sample data
    except:
        m = x0.shape[0]
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j])*(ub[i] - lb[i]) + lb[i]
    return x
#################  The solver function designed for lorenze system  ##################


def solver_lorenz():  # flag = 1 : new point
    var_opt = io.loadmat("allpoints/pre_opt_IC")
    bnd1 = var_opt['lb'][0]
    bnd2 = var_opt['ub'][0]
    n = var_opt['n'][0, 0]
    T_lorenz = var_opt['T_lorenz'][0, 0]
    h = var_opt['h_lorenz'][0, 0]
    user = var_opt['user'][0]
    if n == 1:
        y0 = np.array([23.5712])
    elif n == 3:
        y0 = np.array([23.5712, 23.5712, 23.5712])
    time_method = 1
    DT = 10

    fin = open("allpoints/pts_to_eval.dat", "r")
    flag = read_str(fin.readline().split(), 'i')
    idx = read_str(fin.readline().split(), 'i')
    xm = np.zeros(n)
    for i in range(n):
        xm[i] = read_str(fin.readline().split(), 'f')
    fin.close()

    if flag != 2:
        if flag == 1:  # flag = 1: new point
            T = T_lorenz
            J, zs, ys, xs = lorenz.lorenz_lost2(xm, T, h, bnd2, bnd1, y0, time_method)
        elif flag == 0:  # flag = 0: existing point
            data = io.loadmat("allpoints/pt_to_eval" + str(idx) + ".mat")
            T_zs_lorenz = data['T'][0, 0]
            T = T_zs_lorenz + DT
            J, zs, ys, xs = lorenz.lorenz_lost2(xm, T, h, bnd2, bnd1, y0, time_method, idx)

        fout_surr = open("allpoints/surr_J_new.dat", "w")
        for i in range(zs.shape[0]):
            fout_surr.write(str(zs[i]) + "\n")
        fout_surr.close()

        fout = {'zs': zs, 'ys': ys, 'xs': xs, 'h': h, 'T': T, 'J': J}
        io.savemat("allpoints/pt_to_eval" + str(idx) + ".mat", fout)

        return

    else:  # flag = 2, this is mesh refinement iteration, no function evaluation is performed.

        return
#################  The alpha-DOGS algprithm for lorenz system  ##################


def DOGS_standalone_lorenz_IC():
    '''
    This function reads the set of evaluated points and writes them into the desired file to perform function evaluations
    Note: DOGS_standalone() only exists at the inactivated iterations.
    :return: points that needs to be evaluated
    '''
    # For future debugging, remind that xc and xd generate by DOGS_standalone() is set to be a one dimension row vector.
    # While lb and ub should be a two dimension matrix, i.e. a column vector.
    # The following lines will read input from 'pre_opt_IC' file:
    var_opt = io.loadmat("allpoints/pre_opt_IC")
    n = var_opt['n'][0, 0]
    K = var_opt['K'][0, 0]
    L = var_opt['L'][0, 0]
    Nm = var_opt['Nm'][0, 0]
    bnd2 = var_opt['ub'][0]
    bnd1 = var_opt['lb'][0]
    lb = np.zeros(n)
    ub = np.ones(n)
    user = var_opt['user'][0]
    idx = var_opt['num_point'][0, 0]
    flag = var_opt['flag'][0, 0]
    T_lorenz = var_opt['T_lorenz'][0, 0]
    method = var_opt['inter_par_method']
    xE = var_opt['xE']
    xU = var_opt['xU']
    k = var_opt['iter'][0, 0]
    iter_max = var_opt['iter_max'][0, 0]
    y0 = var_opt['y0'][0]

    if xU.shape[1] == 0:
        xU = xU.reshape(n, 0)

    Data = io.loadmat("allpoints/Yall")
    yE = Data['yE'][0]
    SigmaT = Data['SigmaT'][0]
    T = Data['T'][0]

    # Normalize the bounds of xE and xU
    xE = normalize_bounds(xE, bnd1, bnd2)
    xU = normalize_bounds(xU, bnd1, bnd2)
    # Read the result from 'surr_J_new.dat' file that generated by solver_lorenz:
    if k != 1:

        zs = np.loadtxt("allpoints/surr_J_new.dat")
        J = np.abs(np.mean(zs) - y0)[0]
        one_point = io.loadmat("allpoints/pt_to_eval" + str(idx) + ".mat")
        t = one_point['T']

        xx = uq.data_moving_average(zs, 40).values
        sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx, 18)[0])

        if flag == 1:  # New point

            yE = np.hstack([yE, J])
            SigmaT = np.hstack([SigmaT, sig])
            T = np.hstack([T, T_lorenz])

        elif flag == 0:  # existing point

            yE[idx] = J
            SigmaT[idx] = sig
            T[idx] = t

    #############################################################################
    # The following only for displaying information.
    # NOTICE : Deleting following lines won't cause any affect.
    print('========================  Iteration = ', k, '=======================================')
    print('point to evaluate at this iteration, x = ', xE[:, idx], "flag = ", flag)
    print('==== flag 1 represents new point, 0 represents existed point  =====')
    print('Function Evaluation at this iter: y = ', yE[idx] + SigmaT[idx])
    print('Minimum of all data points(yE + SigmaT): min = ', np.min(yE + SigmaT))
    print('argmin: x_min = ', xE[:, np.argmin(yE + SigmaT)])
    Nm = var_opt['Nm'][0, 0]
    print('Mesh size = ', Nm)
    #############################################################################

    Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
    Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)
    # Calculate the Regression Function
    inter_par = Inter_par(method=method)
    [inter_par, yp] = regressionparametarization(xE, yE, SigmaT, inter_par)
    K0 = 20  # K0 = np.ptp(yE, axis=0)

    # Calculate the discrete function.
    ind_out = np.argmin(yp + SigmaT)
    sd = np.amin((yp, 2 * yE - yp), 0) - L * SigmaT

    ind_min = np.argmin(yp + SigmaT)

    yd = np.amin(sd)
    ind_exist = np.argmin(sd)

    xd = xE[:, ind_exist]

    if ind_min != ind_min:
        # yE[ind_exist] = ((fun(xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist] + 1)
        # T[ind_exist] = T[ind_exist] + 1

        return
    else:

        if SigmaT[ind_exist] < 0.01 * np.ptp(yE, axis=0) * (np.max(ub - lb)) / Nm:
            yd = np.inf

        # Calcuate the unevaluated support points:
        yu = np.zeros([1, xU.shape[1]])
        if xU.shape[1] != 0:
            for ii in range(xU.shape[1]):
                tmp = interpolate_val(xU[:, ii], inter_par) - np.amin(yp)
                yu[0, ii] = tmp / mindis(xU[:, ii], xE)[0]

        if xU.shape[1] != 0 and np.amin(yu) < 0:
            ind = np.argmin(yu)
            xc = np.copy(xU[:, ind])
            yc = -np.inf
            xU = scipy.delete(xU, ind, 1)  # delete the minimum element in xU, which is going to be incorporated in xE
        else:
            while 1:
                xc, yc = tringulation_search_bound_constantK(inter_par, np.hstack([xE, xU]), K * K0, ind_min)
                yc = yc[0, 0]
                if interpolate_val(xc, inter_par) < min(yp):
                    xc = np.round(xc * Nm) / Nm
                    break

                else:
                    xc = np.round(xc * Nm) / Nm
                    if mindis(xc, xE)[0] < 1e-6:
                        break
                    xc, xE, xU, success, _ = points_neighbers_find(xc, xE, xU, Bin, Ain)
                    xc = xc.T[0]
                    if success == 1:
                        break
                    else:
                        yu = np.hstack([yu, (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]])

            if xU.shape[1] != 0:
                tmp = (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]
                if np.amin(yu) < tmp:
                    ind = np.argmin(yu)
                    xc = np.copy(xU[:, ind])
                    yc = -np.inf
                    xU = scipy.delete(xU, ind, 1)  # delete the minimum element in xU, which is incorporated in xE
        # Generate the stop file at this iteration:
        if k + 1 <= iter_max:
            stop = 0
        elif k + 1 > iter_max:
            stop = 1

        fout = open("allpoints/stop.dat", 'w')
        fout.write(str(stop) + "\n")
        fout.close()

        # MESH REFINEMENT ITERATION:
        if mindis(xc, xE)[0] < 1e-6:
            K = 2 * K
            Nm = 2 * Nm
            L += 1
            flag = 2  # flag = 2 represents mesh refinement, in this step we don't have function evaluation.

            # Reconstruct the physical bound of xE and xU
            xE = physical_bounds(xE, bnd1, bnd2)
            xU = physical_bounds(xU, bnd1, bnd2)

            # Store the updated information about iteration to the file 'pre_opt_IC.dat'
            var_opt['K'] = K
            var_opt['Nm'] = Nm
            var_opt['L'] = L
            var_opt['xE'] = xE
            var_opt['xU'] = xU
            var_opt['num_point'] = xE.shape[1] - 1  # Doesn't matter, flag = 2, no function evaluation.
            var_opt['flag'] = flag
            var_opt['iter'] = k + 1
            io.savemat("allpoints/pre_opt_IC", var_opt)

            # Store the function evaluations yE, sigma and time length T:
            data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
            io.savemat("allpoints/Yall", data)

            # Generate the pts_to_eval file for solver_lorenz
            fout = open("allpoints/pts_to_eval.dat", 'w')
            if user == 'Imperial College':
                keywords = ['Awin', 'lambdain', 'fanglein']
                fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
                fout.write(str('IDin') + '=' + str(int(idx)) + "\n")
                for i in range(n):
                    fout.write(str(keywords[i]) + '=' + str(xc[i]) + "\n")
            fout.close()

            return

        if yc < yd:
            if mindis(xc, xE)[0] > 1e-6:

                xE = np.concatenate([xE, xc.reshape(-1, 1)], axis=1)
                flag = 1  # new point
                idx = xE.shape[1] - 1

                # Reconstruct the physical bound of xE and xU
                xE = physical_bounds(xE, bnd1, bnd2)
                xU = physical_bounds(xU, bnd1, bnd2)
                xc = physical_bounds(xc.reshape(-1, 1), bnd1, bnd2)
                xc = xc.T[0]

                # Store the updated information about iteration to the file 'pre_opt_IC.dat'
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = idx
                var_opt['flag'] = flag
                var_opt['iter'] = k + 1
                io.savemat("allpoints/pre_opt_IC", var_opt)

                # Store the function evaluations yE, sigma and time length T:
                data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
                io.savemat("allpoints/Yall", data)

                # Generate the pts_to_eval file for solver_lorenz
                fout = open("allpoints/pts_to_eval.dat", 'w')
                if user == 'Imperial College':
                    keywords = ['Awin', 'lambdain', 'fanglein']
                    fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
                    fout.write(str('IDin') + '=' + str(int(idx)) + "\n")
                    for i in range(n):
                        fout.write(str(keywords[i]) + '=' + str(xc[i]) + "\n")
                fout.close()

                return
        else:
            if mindis(xd, xE)[0] < 1e-10:

                flag = 0  # existing point

                # Reconstruct the physical bound of xE and xU
                xE = physical_bounds(xE, bnd1, bnd2)
                xU = physical_bounds(xU, bnd1, bnd2)
                xd = physical_bounds(xd.reshape(-1, 1), bnd1, bnd2)
                xd = xd.T[0]

                # Store the updated information about iteration to the file 'pre_opt_IC.dat'
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = ind_exist
                var_opt['flag'] = flag
                var_opt['iter'] = k + 1
                io.savemat("allpoints/pre_opt_IC", var_opt)

                # Store the function evaluations yE, sigma and time length T:
                data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
                io.savemat("allpoints/Yall", data)

                # Generate the pts_to_eval file for solver_lorenz
                fout = open("allpoints/pts_to_eval.dat", 'w')
                if user == 'Imperial College':
                    keywords = ['Awin', 'lambdain', 'fanglein']
                    fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
                    fout.write(str('IDin') + '=' + str(int(ind_exist)) + "\n")
                    for i in range(n):
                        fout.write(str(keywords[i]) + '=' + str(xd[i]) + "\n")
                fout.close()

                return
###################################  Delta-DOGS ##############################


def Delta_DOGS_standalone_lorenz():
    '''
    This function reads the set of evaluated points and writes them into the desired file to perform function evaluations
    Note: DOGS_standalone() only exists at the inactivated iterations.
    :return: points that needs to be evaluated
    '''
    # For future debugging, remind that xc and xd generate by DOGS_standalone() is set to be a one dimension row vector.
    # While lb and ub should be a two dimension matrix, i.e. a column vector.
    var_opt = io.loadmat("allpoints/pre_opt")
    n = var_opt['n'][0, 0]
    K = var_opt['K'][0, 0]
    L = var_opt['L'][0, 0]
    Nm = var_opt['Nm'][0, 0]
    bnd2 = var_opt['ub'][0]
    bnd1 = var_opt['lb'][0]
    lb = np.zeros(n)
    ub = np.ones(n)
    user = var_opt['user'][0]
    idx = var_opt['num_point'][0, 0]
    flag = var_opt['flag'][0, 0]
    T_lorenz = var_opt['T_lorenz']
    h = var_opt['h_lorenz']
    method = var_opt['inter_par_method']
    xE = var_opt['xE']
    xU = var_opt['xU']
    if xU.shape[1] == 0:
        xU = xU.reshape(n, 0)

    Data = io.loadmat("allpoints/Yall")
    yE = Data['yE'][0]
    SigmaT = Data['SigmaT'][0]

    Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
    Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

    # TODO FIXME: nff is deleted
    # regret = np.zeros((nff, iter_max))
    # estimate = np.zeros((nff, iter_max))
    # datalength = np.zeros((nff, iter_max))
    # mesh = np.zeros((nff, iter_max))

    inter_par = Inter_par(method=method)
    [inter_par, yp] = regressionparametarization(xE, yE, SigmaT, inter_par)
    K0 = 20# K0 = np.ptp(yE, axis=0)
    # Calculate the discrete function.
    ind_out = np.argmin(yp + SigmaT)
    sd = np.amin((yp, 2 * yE - yp), 0) - L * SigmaT

    ind_min = np.argmin(yp + SigmaT)

    yd = np.amin(sd)
    ind_exist = np.argmin(sd)

    xd = xE[:, ind_exist]

    if ind_min != ind_min:
        # yE[ind_exist] = ((fun(xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist] + 1)
        # T[ind_exist] = T[ind_exist] + 1

        return
    else:

        # if SigmaT[ind_exist] < 0.01 * np.ptp(yE, axis=0) * (np.max(ub - lb)) / Nm:
        #     yd = np.inf

        # Calcuate the unevaluated function:
        yu = np.zeros([1, xU.shape[1]])
        if xU.shape[1] != 0:
            for ii in range(xU.shape[1]):
                tmp = interpolate_val(xU[:, ii], inter_par) - np.amin(yp)
                yu[0, ii] = tmp / mindis(xU[:, ii], xE)[0]

        if xU.shape[1] != 0 and np.amin(yu) < 0:
            t = np.amin(yu)
            ind = np.argmin(yu)
            xc = np.copy(xU[:, ind])
            yc = -np.inf
            xU = scipy.delete(xU, ind, 1)  # create empty array
        else:
            while 1:
                xc, yc = tringulation_search_bound_constantK(inter_par, np.hstack([xE, xU]), K * K0, ind_min)
                yc = yc[0, 0]
                if interpolate_val(xc, inter_par) < min(yp):
                    xc = np.round(xc * Nm) / Nm
                    break

                else:
                    xc = np.round(xc * Nm) / Nm
                    if mindis(xc, xE)[0] < 1e-6:
                        break
                    xc, xE, xU, success, _ = points_neighbers_find(xc, xE, xU, Bin, Ain)
                    xc = xc.T[0]
                    if success == 1:
                        break
                    else:
                        yu = np.hstack([yu, (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]])

            if xU.shape[1] != 0:
                tmp = (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]
                if np.amin(yu) < tmp:
                    ind = np.argmin(yu)
                    xc = np.copy(xU[:, ind])
                    yc = -np.inf
                    xU = scipy.delete(xU, ind, 1)  # create empty array

        if mindis(xc, xE)[0] < 1e-6:
            K = 2 * K
            Nm = 2 * Nm
            L += 1
            flag = 2  # flag = 2 represents mesh refinement, in this step we don't have function evaluation.

            var_opt = {}
            var_opt['n'] = n
            var_opt['K'] = K
            var_opt['Nm'] = Nm
            var_opt['L'] = L
            var_opt['lb'] = bnd1
            var_opt['ub'] = bnd2
            var_opt['user'] = user
            var_opt['inter_par_method'] = method
            var_opt['xE'] = xE
            var_opt['xU'] = xU
            var_opt['num_point'] = xE.shape[1] - 1  # Doesn't matter, flag = 2, no function evaluation.
            var_opt['flag'] = flag
            var_opt['T_lorenz'] = T_lorenz
            var_opt['h_lorenz'] = h
            io.savemat("allpoints/pre_opt", var_opt)

            return

        if yc == yc:
            if mindis(xc, xE)[0] > 1e-6:

                xE = np.concatenate([xE, xc.reshape(-1, 1)], axis=1)
                # xm = lb + (ub - lb) * xc
                flag = 1  # new point

                var_opt = {}
                var_opt['n'] = n
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['lb'] = bnd1
                var_opt['ub'] = bnd2
                var_opt['user'] = user
                var_opt['inter_par_method'] = method
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = xE.shape[1] - 1
                var_opt['flag'] = flag
                var_opt['T_lorenz'] = T_lorenz
                var_opt['h_lorenz'] = h
                io.savemat("allpoints/pre_opt", var_opt)

                return
        else:
            # xm = lb + (ub - lb) * xd
            if mindis(xd, xE)[0] < 1e-10:

                flag = 0  # existing point

                var_opt = {}
                var_opt['n'] = n
                var_opt['K'] = K
                var_opt['Nm'] = Nm
                var_opt['L'] = L
                var_opt['lb'] = bnd1
                var_opt['ub'] = bnd2
                var_opt['user'] = user
                var_opt['inter_par_method'] = method
                var_opt['xE'] = xE
                var_opt['xU'] = xU
                var_opt['num_point'] = ind_exist
                var_opt['flag'] = flag
                var_opt['T_lorenz'] = T_lorenz
                var_opt['h_lorenz'] = h
                io.savemat("allpoints/pre_opt", var_opt)

                return

##########  Initialize function ##########


def Initialize_IC():

    # The following lines are for generate the directory:
    # current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # apts = current_path + "/allpoints"
    # if not os.path.exists(apts):
    #     os.makedirs(apts)

    n = 3  # Dimension of data
    K = 3  # Tuning parameter for continuous search function
    Nm = 8  # Initial mesh grid size
    L = 1  # Tuning parameter for discrete search function
    flag = 1  # Identify
    method = "NPS"  # The strategy for regression function, you can choose NPS or MAPS
    user = 'Imperial College'

    # The following lines represents the initial points:
    # bnd1: lower bounds for physical data
    # bnd2: upper bounds for physical data
    # xE: initial interested points
    # y0: estimate value for minimum
    if n == 1:
        xE = np.array([[0.5, 0.75]])
        y0 = np.array([23.5712])
        bnd2 = np.array([30])
        bnd1 = np.array([24])
    elif n == 2:
        xE = np.array([[0.5, 0.75, 0.5], [0.5, 0.5, 0.75]])
        y0 = np.array([23.5712, 23.5712])
    elif n == 3:
        xE = np.array([[0.5, 0.5, 0.5, 0.75], [0.5, 0.5, 0.75, 0.5], [0.5, 0.75, 0.5, 0.5]])
        y0 = np.array([23.5712, 23.5712, 23.5712])
        bnd2 = np.array([30, 30, 30])
        bnd1 = np.array([24, 24, 24])

    xU = bounds(np.zeros([n, 1]), np.ones([n, 1]), n)

    xE = physical_bounds(xE, bnd1, bnd2)
    xU = physical_bounds(xU, bnd1, bnd2)

    k = 0  # times of iteration, start with 0
    iter_max = 50  # maximum iteration steps
    T_lorenz = 5
    h_lorenz = 0.005
    idx = 0

    var_opt = {}
    var_opt['y0'] = y0
    var_opt['n'] = n
    var_opt['K'] = K
    var_opt['Nm'] = Nm
    var_opt['L'] = L
    var_opt['ub'] = bnd2
    var_opt['lb'] = bnd1
    var_opt['user'] = user
    var_opt['inter_par_method'] = method
    var_opt['xE'] = xE
    var_opt['xU'] = xU
    var_opt['num_point'] = idx
    var_opt['flag'] = flag
    var_opt['T_lorenz'] = T_lorenz
    var_opt['h_lorenz'] = h_lorenz
    var_opt['iter'] = k
    var_opt['iter_max'] = iter_max
    io.savemat("allpoints/pre_opt_IC", var_opt)

    # The following lines are for generating stop file.
    # fout_stop = open("allpoints/stop.dat", 'w')
    # fout_stop.write(str(0) + "\n")
    # fout_stop.close()

    return

