import numpy as np
import scipy.optimize as optimize

class GLAD:
    def __init__(self, Labels):
        self.Labels = Labels.copy()
        self.I, self.J = Labels.shape     # I: number of labelers; J: number of data points
        self.K = 2   # number of labels, only binary is supported
        self._init_paras()

    def _init_paras(self):
        # labeler accuracy
        self.alpha = np.random.randn(self.I)
        # data point difficulty
        self.beta = np.ones((self.J, )) 
    
    def e_step(self):
        self.EZ = np.empty((self.K, self.J))
        for j in xrange(self.J):
            idx = (self.Labels[:, j] != -1)
            Lj = self.Labels[idx, j]
            self.EZ[0, j] = np.prod(bernoulli(logistic(self.alpha[idx] * self.beta[j]), 1-Lj))  
            self.EZ[1, j] = np.prod(bernoulli(logistic(self.alpha[idx] * self.beta[j]), Lj))
        self.EZ /= np.sum(self.EZ, axis=0)
            
    def m_step(self, missing=True, disp=0):
        if missing:
            self.update_theta_batch(disp)
        else:
            self.update_theta_batch_full(disp)
        self.obj = -self.obj_m(np.hstack((self.alpha, np.log(self.beta))))


    def update_theta_batch_full(self, disp):
        def f(theta):
            alpha, beta = theta[:self.I], np.exp(theta[-self.J:])
            tmp = logistic(np.outer(alpha, beta))
            p0 = ((1-self.Labels) * np.log(tmp) + self.Labels * np.log(1-tmp)) * self.EZ[0]
            p1 = (self.Labels * np.log(1-tmp) + (1-self.Labels) * np.log(tmp)) * self.EZ[1]
            return -(p0 + p1).sum()

        def df(theta):
            alpha, beta = theta[:self.I], np.exp(theta[-self.J:])
            tmp = logistic(np.outer(alpha, beta))
            grad_alpha = np.sum((self.Labels * self.EZ[1] + (1-self.Labels) * self.EZ[0] - tmp) * beta, axis=1)
            grad_beta = beta * np.sum(alpha[:,np.newaxis] * (self.Labels * self.EZ[1] + (1-self.Labels) * self.EZ[0] - tmp), axis=0)
            return -np.hstack((grad_alpha, grad_beta))
        
        theta0 = np.hstack((self.alpha, np.log(self.beta)))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(self.obj_m(theta_hat), d['task'])
            else:
                print 'f={}, {}'.format(self.obj_m(theta_hat), d['warnflag'])
            app_grad = approx_grad(self.obj_m, theta_hat)
            for i in xrange(self.I):
                print 'alpha[{:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(i, theta_hat[i], app_grad[i], df(theta_hat)[i], np.abs(app_grad[i] - df(theta_hat)[i]))
            for j in xrange(self.I, self.I + self.J):
                print 'beta[{:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(j-self.I, np.exp(theta_hat[j]), app_grad[j], df(theta_hat)[j], np.abs(app_grad[j] - df(theta_hat)[j]))
        self.alpha, self.beta = theta_hat[:self.I], np.exp(theta_hat[-self.J:])


    def update_theta_batch(self, disp):
        def df(theta):
            alpha, beta = theta[:self.I], np.exp(theta[-self.J:])
            grad_alpha = np.zeros((self.I,))
            grad_beta = np.zeros((self.J,))
            for j in xrange(self.J):
                idx = (self.Labels[:, j] != -1)
                Lj = self.Labels[idx, j]
                tmp = logistic(alpha[idx] * beta[j])
                grad_alpha[idx] += beta[j] * (Lj * self.EZ[1,j] + (1-Lj) * self.EZ[0,j] - tmp)
                grad_beta[j] = beta[j] * np.sum(alpha[idx] * (Lj * self.EZ[1,j] + (1-Lj) * self.EZ[0,j] - tmp))
            return -np.hstack((grad_alpha, grad_beta))

        theta0 = np.hstack((self.alpha, np.log(self.beta)))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(self.obj_m, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(self.obj_m(theta_hat), d['task'])
            else:
                print 'f={}, {}'.format(self.obj_m(theta_hat), d['warnflag'])
            app_grad = approx_grad(self.obj_m, theta_hat)
            for i in xrange(self.I):
                print 'alpha[{:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(i, theta_hat[i], app_grad[i], df(theta_hat)[i], np.abs(app_grad[i] - df(theta_hat)[i]))
            for j in xrange(self.I, self.I + self.J):
                print 'beta[{:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(j-self.I, np.exp(theta_hat[j]), app_grad[j], df(theta_hat)[j], np.abs(app_grad[j] - df(theta_hat)[j]))
        self.alpha, self.beta = theta_hat[:self.I], np.exp(theta_hat[-self.J:])

    def obj_m(self, theta):
        alpha, beta = theta[:self.I], np.exp(theta[-self.J:])
        obj = 0
        for j in xrange(self.J):
            idx = (self.Labels[:, j] != -1)
            Lj = self.Labels[idx, j]
            tmp = logistic(alpha[idx] * beta[j])
            obj += np.sum(((1-Lj) * np.log(tmp) + Lj * np.log(1-tmp)) * self.EZ[0, j])
            obj += np.sum((Lj * np.log(1-tmp) + (1-Lj) * np.log(tmp)) * self.EZ[1, j])
        return -obj

def logistic(x):
    return 1./(1 + np.exp(-x))

def bernoulli(p, l):
    return p ** l * (1-p) ** (1-l)

def approx_grad(f, x, delta=1e-8):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i]) - f(x - diff[i])) / (2*delta)
    return grad
