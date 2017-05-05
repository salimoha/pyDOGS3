import numpy as np
import os, inspect
import dogs, uq
import scipy.io as io
from pathlib import Path
import tr
import shutil
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
        bnd2 = np.array([30])
        bnd1 = np.array([24])

    elif n == 2:
        xE = np.array([[0.5, 0.75, 0.5], [0.5, 0.5, 0.75]])

    elif n == 3:
        xE = np.array([[0.5, 0.5, 0.5, 0.75], [0.5, 0.5, 0.75, 0.5], [0.5, 0.75, 0.5, 0.5]])
        bnd2 = np.array([30, 30, 30])
        bnd1 = np.array([24, 24, 24])
    
    xU = dogs.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)

    xE = dogs.physical_bounds(xE, bnd1, bnd2)
    xU = dogs.physical_bounds(xU, bnd1, bnd2)
    
    k = 0  # times of iteration, start with 0
    iter_max = 500  # maximum iteration steps
    idx = 0
    
    var_opt = {}
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
    var_opt['iter'] = k
    var_opt['iter_max'] = iter_max
    io.savemat("allpoints/pre_opt_IC", var_opt)
    
    return


def DOGS_standlone():
    print('==================================================')

    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    pre_opt = current_path + "/allpoints/pre_opt_IC.mat"
    pre_opt_path = Path(pre_opt)

    pre_Y = current_path + "/allpoints/Yall.mat"
    pre_Y_path = Path(pre_Y)

    # Check whether or not it is the first iteraiton, if the optimizaton information file pre_opt_IC doesn't exist, then
    # generate that file:
    if not pre_opt_path.is_file():
        Initialize_IC()

    var_opt = io.loadmat("allpoints/pre_opt_IC")
    k = var_opt['iter'][0, 0]


    # initilizaiton
    if k == 0:  # k is the number of iteration. k = 0 means that the initialization is not finished.

        if pre_Y_path.is_file():  # The file 'Yall' exists.

            data = io.loadmat("allpoints/Yall")
            # read the previous points
            yE = data['yE'][0]
            SigmaT = data['SigmaT'][0]
            T = data['T'][0]

            zs = np.loadtxt("allpoints/surr_J_new.dat")

            xx = uq.data_moving_average(zs, 40).values
            ind = tr.transient_removal(xx)
            sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx[ind:], 18)[0])
            t = len(zs)
            J = np.abs(np.mean(xx[ind:]))

            yE = np.hstack((yE, J))
            SigmaT = np.hstack((SigmaT, sig))
            T = np.hstack((T, t))
            data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
            io.savemat("allpoints/Yall", data)

            print(' len of yE = ', len(yE))
        else:
            pre_J = current_path + "/allpoints/surr_J_new.dat"
            pre_J_path = Path(pre_J)
            if not pre_J_path.is_file():
                # The very first iteration, the file 'Yall' doesn't exist.
                yE = np.array([])
                print("The first time running the iteration")
                print(' len of yE = ', len(yE))
                print('iter k = ', k)
            else:
                # The second time of running the algorithm.

                yE = np.array([])
                SigmaT = np.array([])
                T = np.array([])

                # Read from surr_J_new.
                zs = np.loadtxt("allpoints/surr_J_new.dat")

                xx = uq.data_moving_average(zs, 40).values
                ind = tr.transient_removal(xx)
                sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx, 18)[0])
                t = len(zs)
                J = np.abs(np.mean(xx[ind:]))

                yE = np.hstack((yE, J))
                SigmaT = np.hstack((SigmaT, sig))
                T = np.hstack((T, t))
                data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
                io.savemat("allpoints/Yall", data)

                print("The second time running the iteration")
                print(' len of yE = ', len(yE))
                print('iter k = ', k)
                print('function evaluation at this iteration: ', J)

        # we read pre_opt_IC.mat
        # untill len(yE) < n+1 then k=1
        # var_opt = io.loadmat("allpoints/pre_opt_IC")
        # The following variables are needed for initialization:

        xE = var_opt['xE']
        n = var_opt['n'][0, 0]
        flag = var_opt['flag'][0, 0]

        if len(yE) < n + 1:
            # Generate the point that we want to evaluate.
            xcurr = np.copy(xE[:, len(yE)])
            fout = open("allpoints/pts_to_eval.dat", 'w')
            keywords = ['Awin', 'lambdain', 'fanglein']
            fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
            fout.write(str('IDin') + '=' + str(int(len(yE))) + "\n")
            for j in range(n):
                fout.write(keywords[j] + '=' + str(float(xcurr[j])) + "\n")
            fout.close()

            var_opt['iter'] = 0
            var_opt['num_point'] = len(yE)
            io.savemat("allpoints/pre_opt_IC", var_opt)

            print('point to eval at this iteration x = ', xcurr)
            print('len of yE = ', len(yE))
            return
        else:

            # Initialization complete
            var_opt['iter'] = 1

            io.savemat("allpoints/pre_opt_IC", var_opt)

            # The following are just for displaying:
            print('==================================================')
            print('==================================================')
            print('iter k = ', var_opt['iter'])
            r = input('Initializtion complete, type anything to continue: ')

            # Run one iteration after initialization.
            dogs.DOGS_standalone_IC()

            return

    else:

        dogs.DOGS_standalone_IC()
        return



def run_opti():
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # generate the directory
    apts = current_path + "/allpoints"
    if not os.path.exists(apts):
        os.makedirs(apts)
    # create the stop file
    stp = current_path + "allpoints/stop.dat"
    stp_path = Path(stp)
    if not stp_path.is_file():
        stop = 0
        fout = open("allpoints/stop.dat", 'w')
        fout.write(str(stop) + "\n")
        fout.close()

    while stop == 0:

        # DOGS_standalone() will first generate the initial points, then run alpha-DOGS algorithm.
        DOGS_standlone()

        # The "solver_lorenz" function will read from "pts_to_eval" and generate a file named "surr_J_new.dat"
        # containing all information about function evaluation.
        dogs.solver_lorenz()

        # Read stop file
        stop = int(np.loadtxt("allpoints/stop.dat"))
        print('stop = ', stop)
    return
################################################################################################################
run_opti()
DOGS_standlone()

# Delete the directory of allpoints
current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
shutil.rmtree(current_path + "/allpoints")

