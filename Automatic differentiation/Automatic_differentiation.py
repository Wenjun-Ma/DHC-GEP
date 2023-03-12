from tkinter.messagebox import NO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import scipy.io as scio


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, y, t, rho, layers):
        
        X = np.concatenate([y, t], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.y = X[:,0:1]
        self.t = X[:,1:2]

        self.rho = rho
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])

        
        self.rho_pred = self.net_NS(self.y_tf, self.t_tf)
        self.fdiff_pred = self.net_diff(self.y_tf, self.t_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.rho_tf - self.rho_pred))
            
                    
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, y, t):
        
        w = self.neural_net(tf.concat([y,t], 1), self.weights, self.biases)
        
        rho = w[:,0:1]
        
        return rho
    
    def net_diff(self, y, t):
        w = self.neural_net(tf.concat([y,t], 1), self.weights, self.biases)
        
        rho = w[:,0:1]
        
        rho_t = tf.gradients(rho, t)[0] 
        rho_y = tf.gradients(rho, y)[0]
        rho_yy = tf.gradients(rho_y, y)[0]
        rho_3y = tf.gradients(rho_yy, y)[0]

        return rho_t, rho_y, rho_yy, rho_3y
        
    def callback(self, loss):
        print('Loss: %.3e.' % (loss))
    
    def train(self, nIter): 

        tf_dict = {self.y_tf: self.y, self.t_tf: self.t, self.rho_tf: self.rho}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],#
                                loss_callback = self.callback)
            
    
    def predict(self, y_star, t_star):
        
        tf_dict = {self.y_tf: y_star, self.t_tf: t_star}

        rho_star = self.sess.run(self.rho_pred, tf_dict)
        f_diff_star = self.sess.run(self.fdiff_pred, tf_dict)
        
        return rho_star, f_diff_star

def plot_solution(X_star, u_star, index):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
if __name__ == "__main__": 
      
    N_train = 20000
    
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # Load Data
    lamada_Ar = 6.2497e-8
    tau_Ar = 1.644e-10
    y = (np.linspace(0, 100, 256, endpoint=False))* lamada_Ar
    t = np.linspace(0, 1000, 101) * tau_Ar
    
    iter_times = 20000
    rho = np.loadtxt(f'../data/density.dat', delimiter=' ').reshape((256,101))
    
    N = y.shape[0]
    T = t.shape[0]
    
    YY = np.tile(y[:,None], (1,T)) # N x T
    TT = np.tile(t[:,None], (1,N)).T # N x T

    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    rho = rho.flatten()[:,None] # NT x 1
    
    model = PhysicsInformedNN(y, t, rho, layers)
    model.train(iter_times)

    # Prediction
    rho_pred, fdiff_pred = model.predict(y, t)

    print(np.array(fdiff_pred).shape)
    
    scio.savemat(f'../data/Diffusion_flow_new.mat', {'rho':rho, 'rho_t':fdiff_pred[0], 'rho_y':fdiff_pred[1], 'rho_yy':fdiff_pred[2], 'rho_3y':fdiff_pred[3]})

    # Error
    error_rho = np.linalg.norm(rho-rho_pred,2)/np.linalg.norm(rho,2)
    a =  (rho-rho_pred)/rho
    abs_error_rho = np.mean(abs(a))
 
    print('Error rho: %e' % (error_rho)) 
    print('Abs error rho: %e' % (abs_error_rho))
            
    

    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    