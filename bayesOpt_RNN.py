import numpy as np
import bayesopt as bo
import RNN_train_wrapper as rtw
# from bayesoptmodule import BayesOptDiscrete

params = {}
# params['n_iterations'] = 50
# params['n_iter_relearn'] = 5
# params['n_init_samples'] = 2
dim=4                     # n dimensions
# lb = np.array([500.0,100.0,100.0,100.0])
# ub = np.array([2000.0,512.0,256.0,1000.0])
lb = np.ones((dim,))*0
ub = np.ones((dim,))*20

print "Callback implementation"

mvalue, x_out, error = bo.optimize(rtw.wrapper, dim, lb, ub, params)

print "Result", mvalue, "at", x_out