import sklearn as sk
import numpy as np
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import filterpy.kalman as KF

def non_linearF(xk,dt = 0.5):
    """
    Non-linear transition function f
    :param xk:
     a 1x 2 matrix. xk[0,0]: position. xk[0,1]: velocity
    :return:
        return estimate state vector
    """
    x_k1 = np.mat(xk)
    if x_k1.shape[1]>=2:
        a =0.3
        b = -1.4
        c = 0
        # position update
        x_k1[0, 0] = x_k1[0,0] + x_k1[0,1] *dt
        # velcoity update
        x_k1[0, 1] = a * x_k1[0,1]**2 + b * x_k1[0,1] +c
        # x_k1[0, 1] = np.exp(np.log(x_k1[0,1]) + a)
        pass
    else:
        print("Unvalid input state")

    return x_k1

def dF(xk):
    """
    The corresponding derivation of non-linear transiton function f at xk state
    :param xk:
    :return:
        return transition matrix F = [[df/dvt, 0]
                                      [0,       df/dst] ]
        df/dvt =2avt + b
        df/dst = 1
    """
    a= 0.3
    b = -1.4
    xk = np.mat(xk)
    F = np.mat(np.identity(xk.shape[1]))
    F[0,0] = 2.0 * a * xk[0,1] +b
    # F[0,0] = np.exp(np.log(xk[0,1]) + a)/float(xk[0,1])
    F[1,1] = 1
    return  F

def loadData(model = "linear", f= None, df= None):
    """
    x : time array
    y : matrix of states
    state vector: [x1, x1', x2,x2']
    :return:
    """
    features_num =3
    dt = 0.5
    t = np.arange(1, 50.0,dt,dtype= float)
    y = np.mat(np.zeros([len(t), features_num]))
    H = np.mat(np.identity(features_num))
    # linear input examples
    if model.lower() == 'linear':
        a0= 1.2
        y[:,0] = np.mat( a0 *t**2/2).T
        y[:,1] = np.mat(a0 * t).T
        y[:,2] =np.mat(np.ones([1,len(t)]) * a0).T

        phi =np.mat([[1,dt,0],
                     [0,1,0],
                     [0,1,0]])
    elif model.lower() == 'accmove':
        a0 = 1.2
        k0 = 1.1
        # movement model
        # formulas:
        #  a(t) = k0^t *a0
        #  y = s(t) = 1/2 *a(t)*t^2
        # position
        y[:,0] =((a0 * np.power(np.ones([1,len(t)])*k0,t) * t**2)/2).T
        # velocity
        y[:,1] = (np.power(np.ones([1,len(t)])*k0,t)* a0 * t).T
        # acceleration
        y[:,2] = (np.power(np.ones([1,len(t)])*k0,t)* a0).T
        phi = np.mat([[1, dt, dt ** 2 * k0],
                      [0, 1, k0 * dt],
                      [0, 0, k0]])
        # generate the corresponding state transition matrix
        # and connection matrix
    elif model.lower() == 'accosc':
        # acceleration-oscillating model
        a0 = 2.0*np.array(np.sin(t))
        # print("a0:",a0)
        # movement model
        # formulas:
        #  a(t) = k0^t *a0
        #  y = s(t) = 1/2 *a(t)*t^2
        # position
        y[:, 0] = ((a0 * t**2) / 2).reshape([1,98]).T
        # velocity
        y[:, 1] = (a0 * t).reshape([1,98]).T
        # acceleration
        y[:, 2] = a0.reshape([1,98]).T
        phi = np.mat([[1, dt, dt ** 2 ],
                      [0, 1, dt ],
                      [0, 0, 1]])
        pass
    else:
        # non-linear
        features_num = 2
        y = np.mat(np.zeros([len(t), features_num]))
        H = np.mat(np.identity(features_num))
        # initialize velocity v=1.0. position =0 with non-linear function
        y[0,:] = f(np.mat([0,0.4]))
        for i in range(1, len(t)-1):
            # input matrix
            y[i,:] = f(y[i-1,:])
            pass
        #initial phi
        phi = df(y[0,:])
        print("Phi: ",phi)
    return t, y, H, phi




def plot_data(x,y, estmated_y,actual_obv):
    s = { 0:"Position",1:"Velocity",2:"acceleration"}
    for i in range(y.shape[1]):
        plot = plt.figure()
        ax = plot.add_subplot(111)
        ax.scatter(x[:],actual_obv[:,i].flatten()[0],s=10,c='g')
        ax.scatter(x[:],y[:,i].flatten()[0],s=10,c='r')
        ax.scatter(x[:],estmated_y[:,i].flatten()[0],s=10,c='b')
        ax.set_ylabel(s[i])
        ax.set_xlabel("time")
        plt.show()
    pass

def generate_Noise(shape=4, model="Gauss", sigma =1.0):
    R = np.mat(np.identity(shape))
    Q = np.mat(np.identity(shape))
    from filterpy.common import Q_discrete_white_noise
    if model.lower() == "gauss":

        for i in range(shape):
            R[i,i] = np.mat(np.random.normal(0,sigma,1))
            Q[i,i] = np.mat(np.random.normal(0, sigma,1))


    elif model.lower() == "linear":
        R = np.mat(np.identity(shape) * sigma)
        Q = np.mat(np.identity(shape) * sigma)
    else:
        print("Don't recognize model:%s"%(model))
    return R, Q




def EKFTest():
    # load test data
    x, y, H, phi = loadData(model="accmove",f= non_linearF, df= dF)

    # add noise term to expectation output to obtain actual observation
    actual_obv = np.mat(np.zeros([y.shape[0], y.shape[1]]))
    for i in range(y.shape[1]):
        actual_obv[:, i] = y[:, i] + np.mat(np.random.normal(0, 100, y.shape[0])).T

    print("x:", x)
    print("y:", y)
    print("actual:", actual_obv.shape)
    # generate noise model
    R, Q = generate_Noise(3, model="linear", sigma=0.01)
    print("R:", R)
    print("Q:", Q)

    kf_obj = KF.KalmanFilter(dim_x=y.shape[1],dim_z=y.shape[1] )
    kf_obj.H = H
    kf_obj.F = phi
    kf_obj.Q = Q
    kf_obj.R = R
    kf_obj.z = actual_obv
    estimated_v = []
    # predict data
    for k in range(y.shape[0]):
        # input the control state vector
        if k + 1 >= y.shape[0]:
            i = k
        else:
            i = k + 1
        t = x[i]
        # Kalman filter
        kf_obj.x = y[i,:].T
        kf_obj.update(z=actual_obv[k,:].T)
        kf_obj.predict()
        estimated_v.append(kf_obj.x_prior)
        # print("estimate:",estimated_v[-1])
        pass

    # plot data
    # estimated_v : filtered prediction output
    # y: expected output
    # actual_obv: actual observation data
    plot_data(x, y, np.mat(np.array(estimated_v)), actual_obv)

    # print(x,np.array(estimated_v))



if __name__ == "__main__":
    print("Kalman Filter Demo")
    # KFTest()
    EKFTest()
    pass
