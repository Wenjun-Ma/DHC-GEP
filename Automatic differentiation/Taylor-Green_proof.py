import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib import cm

mpl.rc('text', usetex=True)
mpl.rc('font', family='Arial')
mpl.rc('axes', titlesize=20)
if __name__ == "__main__":

    train_data = scio.loadmat('../data/Taylor_vortex_flow.mat')
    u = train_data['u'].reshape(32,32,32)
    v = train_data['v'].reshape(32,32,32)
    vor = train_data['vor'].reshape(32,32,32)

    u_x = train_data['u_x'].reshape(32,32,32)
    u_xx = train_data['u_xx'].reshape(32,32,32)
    u_y = train_data['u_y'].reshape(32,32,32)
    u_yy = train_data['u_yy'].reshape(32,32,32)

    v_x = train_data['v_x'].reshape(32,32,32)
    v_xx = train_data['v_xx'].reshape(32,32,32)
    v_y = train_data['v_y'].reshape(32,32,32)
    v_yy = train_data['v_yy'].reshape(32,32,32)

    vor_x = train_data['vor_x'].reshape(32,32,32)
    vor_xx = train_data['vor_xx'].reshape(32,32,32)
    vor_y = train_data['vor_y'].reshape(32,32,32)
    vor_yy = train_data['vor_yy'].reshape(32,32,32)

    y_list = range(1,33)
    new_y_list = [(y-0.5)/32*50 for y in y_list]
    x = y = new_y_list
    T = 20
    Z = vor_xx[:,:,T]
    Z_theory = -vor[:,:,T]

    figsize = 13,10
    fig, ax = plt.subplots(figsize=figsize)
    X, Y = np.meshgrid(x, y)
    cs = ax.contourf(X, Y, Z, levels = np.linspace(-60, 60, 13,endpoint = True), cmap=cm.OrRd)
    ax.contour(X, Y, Z_theory, levels=cs.levels, colors='k') 
    cbar = fig.colorbar(cs)
    cbar.ax.tick_params(labelsize=22)

    Z = Z.reshape(-1,1)
    Z_theory = Z_theory.reshape(-1,1)
    relative_error = np.linalg.norm(Z-Z_theory, 2)/np.linalg.norm(Z_theory, 2)
    
    
    font1 = {'family' :  'Arial','weight' : 'normal','size'   : 20,}

    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname( 'Arial') for label in labels]

    font2 = {'family' :  'Arial','weight' : 'normal','size'   : 30,}
    plt.xlabel('$x/\\lambda $', font2)
    plt.ylabel('$ y/\\lambda  $', font2)
    plt.subplots_adjust(left = 0.2,bottom=0.128)

    plt.xlim((0, 50))
    plt.ylim((0, 50))
    plt.show()
    print(relative_error)
