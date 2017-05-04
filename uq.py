import numpy as np
import math
from scipy import optimize

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd

pd.options.display.max_rows = 20;
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
Authors: Shahrouzalimo & KimuKook
Modified: Feb. 2017


If you use this code please cite:
Beyhaghi, P., Alimohammadi, S., and Bewley, T., A multiscale, asymptotically unbiased approach to uncertainty quantification
in the numerical approximation of infinite time-averaged statistics. Submitted to Journal of Uncertainity Quantification.
'''


########################## uncertianty quantication for time-averaging error ############################
def emprical_sigma(x, s):
    N = len(x)
    sigma = np.zeros(len(s))
    for jj in range(len(s)):
        mu = np.zeros([N // s[jj]])
        for i in range(N // s[jj]):
            inds = np.arange(i * s[jj], (i + 1) * s[jj])
            mu[i] = (x[np.unravel_index(inds, x.shape, 'F')]).mean(axis=0)
        sigma[jj] = (mu ** 2).mean(axis=0)
    return sigma


def stationary_statistical_learning_reduced(x, m):
    N = len(x)
    M = math.floor(math.sqrt(N))
    s = np.arange(1, 2 * M + 1)
    variance = x.var(axis=0) * len(x) / (len(x) - 1)
    x = np.divide(x - x.mean(axis=0), ((x.var(axis=0)) * len(x) / (len(x) - 1)) ** (1 / 2))

    sigmac2 = emprical_sigma(x, s)
    tau = np.arange(1, m + 1) / (m + 1)

    fun = lambda tau: Loss_fun_reduced(tau, sigmac2, 1)
    jac = lambda tau: Loss_fun_reduced(tau, sigmac2, 2)
    theta = np.zeros([2 * m])
    bnds = tuple([(0, 1) for i in range(int(m))])
    opt = {'disp': False}
    res_con = optimize.minimize(fun, tau, jac=jac, method='L-BFGS-B', bounds=bnds, options=opt)
    theta_tau = np.copy(res_con.x)

    theta_A = optimum_A(theta_tau, sigmac2)
    theta = np.concatenate((theta_A, theta_tau), axis=0)
    moment2_model, corr_model = Thoe_moment2(np.concatenate((np.array([1]), np.array([0]), theta), axis=0), N)

    sigma2_N = moment2_model[-1] * variance
    print("sigma2N = ", sigma2_N)
    return sigma2_N, theta, moment2_model


def Loss_fun_reduced(tau, sigmac2, num_arg):
    #   This function minimizes the linear part of the loss function that is a least square fit for the varaince of time averaging errror
    #   This is done using alternative manimization as fixing the tau value fixed.
    #   Autocorrelation function is rho = A_1 tau_1^k + ... +A_m tau_m^k
    #   If num_arg == 1, return L
    #   If num_arg == 2, return DL
    #   If num_arg == 3, return L,DL
    L = np.array([0])
    m = len(tau)
    H = np.zeros([m, m])
    Ls = np.zeros([m])
    DL = np.zeros([m])
    for ss in range(1, len(sigmac2) + 1):
        for ii in range(m):
            as_ = np.arange(1, ss + 1)
            Ls[ii] = 1 / ss * (1 + 2 * np.dot(1 - np.divide(as_, ss), tau[ii] ** as_.reshape(-1, 1))) - sigmac2[ss - 1]
            DL[ii] = 2 * Ls[ii] * 2 / ss * (
                np.dot((as_ - np.power(as_, 2) / ss), np.power(tau[ii], (as_ - 1).reshape(-1, 1))))
        H = H + np.multiply(Ls.reshape(-1, 1), Ls)

    fun = lambda x: 0.5 * np.dot(x.T, np.dot(H, x))
    jac = lambda x: np.dot(x.T, H)
    Aineq = np.identity(m)
    bineq = np.zeros([m])
    Aeq = np.ones([1, m])
    beq = np.ones([1])
    cons = ({'type': 'ineq', 'fun': lambda x: bineq + np.dot(Aineq, x), 'jac': lambda x: Aineq},
            {'type': 'eq', 'fun': lambda x: beq - np.dot(Aeq, x), 'jac': lambda x: -Aeq})
    x0 = np.ones([m, 1]) / m
    res = optimize.minimize(fun, x0, jac=jac, constraints=cons, method='SLSQP')
    L = res.fun
    DL = np.multiply(DL, res.x)
    if num_arg == 1:
        return L
    elif num_arg == 2:
        return DL
    else:
        return L, DL


def optimum_A(tau, sigmac2):
    # tau need to be a vector
    m = len(tau)
    H = np.zeros([m, m])
    Ls = np.zeros([m])
    DL = np.zeros([m])
    for ss in range(1, len(sigmac2) + 1):
        for ii in range(m):
            as_ = np.arange(1, ss + 1)
            Ls[ii] = 1 / ss * (1 + 2 * np.dot(1 - np.divide(as_, ss), tau[ii] ** as_.reshape(-1, 1))) - sigmac2[ss - 1]
            DL[ii] = 2 * Ls[ii] * 2 / ss * (
                np.dot((as_ - np.power(as_, 2) / ss), np.power(tau[ii], (as_ - 1).reshape(-1, 1))))
        H = H + np.multiply(Ls.reshape(-1, 1), Ls)
    c = np.zeros([m])
    A_ineq = np.identity(m)
    b_ineq = np.zeros([m])
    A_eq = np.ones([1, m])
    b_eq = np.array([1])
    x0 = np.ones([m, 1]) / m
    func = lambda x: 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(c, x)
    jaco = lambda x: np.dot(x.T, H) + c
    cons = ({'type': 'ineq',
             'fun': lambda x: b_ineq + np.dot(A_ineq, x),
             'jac': lambda x: A_ineq},
            {'type': 'eq',
             'fun': lambda x: b_eq - np.dot(A_eq, x),
             'jac': lambda x: A_eq})
    opt = {'disp': False}
    A = optimize.minimize(func, x0, jac=jaco, constraints=cons, method='SLSQP', options=opt)
    return A.x


def Thoe_moment2(theta, N):
    # Equaiton (?) in the paper
    m = int(len(theta) / 2 - 1)
    corr_model = exponential_correlation(theta[2:m + 2], theta[m + 2:], N)
    s = np.arange(1, N + 1).reshape(-1, 1)
    moment2 = theta[1] + theoritical_sigma(corr_model.reshape(-1, 1), s, theta[0])
    return moment2, corr_model


def exponential_correlation(A, tau, N):
    # exponential correlation model
    # sum A_i tau^k
    corr = np.zeros([1, N])
    for ii in range(1, len(tau) + 1):
        corr = corr + A[np.unravel_index(ii - 1, A.shape, 'F')] * np.power(tau[ii - 1], np.arange(1, N + 1))
    return corr


def theoritical_sigma(corr, s, sigma02):
    sigma = np.zeros([len(s)])
    for ii in range(int(len(s))):
        sigma[ii] = 1.0
        for jj in range(1, int(s[ii])):
            sigma[ii] = sigma[ii] + 2 * (1 - jj / s[ii]) * corr[jj - 1]
        sigma[ii] = sigma02 * sigma[ii] / s[ii]
    return sigma


################################## transient detector ##################################


def transient_removal(x=[]):
    # # transient_removal(x) is an automatic procedure to determine the nonstationary part a signal from the stationary part.
    #  It finds the transient time of the simulation using the minimum variance intreval.
    #  INPUT:
    #  x: is the signal which after some transient part the signal becomes stationary
    #  OUTPUT:
    #  ind: is the index of signal that after that the signal could be considered as a stationry signal.

    N = len(x)
    k = np.int_(N / 2)
    y = np.zeros((k, 1))
    for kk in np.arange(k):
        y[kk] = np.var(x[kk + 1:]) * 1.0 / (N - kk - 1.0)
    y = np.array(-y)
    ind = np.argmax(y)

    return ind


def data_moving_average(ym, mm=40):
    '''
    reducing the size of data. Since it is stationary its std and mean do not change
    :param ym: original signal
    :param mm: moving block average
    :return: new signal with length of len(ym)/mm
    '''
    #
    NN = len(ym);
    ym2 = ym[:int(math.floor(NN / mm) * mm)]
    #     print(ym2)
    x = pd.DataFrame(np.reshape(ym2, (math.floor(NN / mm), mm)))
    y = np.mean(x, axis=1)
    #     print(y)
    return y


def readInputFile(filePath):
    #    retVal = []
    #    with open(filePath, 'rb') as csvfile:
    #        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #        for row in filereader:
    #            retVal.append([int(row[0]), int(row[1]), int(row[2])])

    # Example: x = readInputFile(data1FilePath)

    retVal = []
    with open(filePath) as file:
        line = file.readline()
        arr = [float(a) for a in line.split(',')]
        #        retVal.append(file.readline())
        retVal.append(arr)
    return retVal[0]


def transient_drag(x, Safety=13):
    #    an ad-hoc approach for detecting transient based on the moving blocking (we are not using this approach.)
    index = 0;
    IND = 0;
    INDEX = 0
    viol_safe = 0;
    xc = np.copy(x)
    for idx in range(100, len(x), 2000):
        index = transient_removal(x[IND:idx + IND])
        index0 = np.copy(index)
        IND = IND + int(index)

        if index < 5:
            viol_safe = viol_safe + 1
            #             continue
            if viol_safe >= Safety:
                xc = x[int(INDEX):]
                print('index of transient start', INDEX)
                return INDEX, xc
            else:
                if index0 != 0:
                    INDEX = IND + index

                else:
                    index = 100;
                    IND = IND + index
                    INDEX = np.copy(IND)


def relax_data(xc, numSTD=6):
    #     x = pd.Series(xc,index='Drag')
    thershold = np.median(xc) + np.std(xc) * numSTD
    idt = xc > thershold
    xc[idt] = np.nan
    idt2 = xc < -thershold
    xc[idt2] = np.nan
    return xc


def UQ_model_Richardson_stationary(C0, sigma0, T, N, p=3.0):
    # Uncertaitniy estimation
    # Detailed explanation goes here
    # for now we consider the staiotnary IID model for sigma_T
    sigma_T = sigma0 / np.sqrt(T)

    # TRANSIENT DETECTOR
    # STATIONARY PART UQ
    #NumCorrParam=18
    #sigma_T = stationary_statistical_learning_reduced(yE,NumCorrParam)

    h = 1.0/N # cosidering domain with length 1;
    sigma_h = C0*h**p
    # if (T.shape[0]) > 1:
    #     for ii in range(len(sigma_h)):
    #         if (sigma_h[ii]>10):
    #             sigma_h = 1
    # elif T.shape[0]==1:
    if (sigma_h > 10):
        sigma_h = 1


    sigma = np.add(sigma_h, sigma_T)
    return(sigma)



