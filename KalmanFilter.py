import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, std_acc, x_std_meas, y_std_meas):
        #Sampling Time
        self.dt = dt

        #Control Input Variables
        self.u = np.matrix([[u_x],[u_y]])

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])

        #Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        #Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Transformation Matrix Ct
        self.Ct = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Process Noise Covariance
        self.Rt = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Measurement Noise Covariance
        self.Qt = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        #Initial Covariance Matrix
        self.Sigma_T = np.eye(self.A.shape[1])

    def predict(self):
        #Intial state updated
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        #Error Covariance Updated
        self.P = np.dot(np.dot(self.A, self.Sigma_T), self.A.T) + self.Rt
        return self.x[0:2]

    def update(self, z): # z is sensor measured value
      
        S = np.dot(self.Ct, np.dot(self.Sigma_T, self.Ct.T)) + self.Qt

       #Kalman Gain K 
        K = np.dot(np.dot(self.Sigma_T, self.Ct.T), np.linalg.inv(S))  #ERt.(11)
        
        #Manipulating Mean by K and z
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.Ct, self.x))))   #ERt.(12)
        
        #Identity Matrix I
        I = np.eye(self.Ct.shape[1])

        # New Covariance of Posterior Believe
        self.P = (I - (K * self.Ct)) * self.P   #ERt.(13)
        return self.x[0:2]