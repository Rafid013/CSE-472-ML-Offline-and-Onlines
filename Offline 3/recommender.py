import numpy as np
import random as rand
import math


class Recommender:
    def __init__(self, lambda_u, lambda_v, K, threshold):
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.K = K
        self.u_T = []
        self.v = []
        self.threshold = threshold

    def train(self, data):
        u_T = self.u_T
        v = self.v
        N = data.shape[0]
        M = data.shape[1]
        for i in range(N):
            temp = []
            for k in range(self.K):
                temp.append(rand.uniform(-10, 10))
            u_T.append(temp)

        u_T = np.matrix(u_T)

        for i in range(M):
            temp = []
            for k in range(self.K):
                temp.append(rand.uniform(-10, 10))
            v.append(temp)

        v = np.matrix(v).transpose()

        prev_L_reg = -100

        while True:
            for m in range(M):
                temp1 = np.matrix(np.zeros((self.K, self.K), dtype=np.float32))
                temp2 = np.matrix(np.zeros((self.K, 1), dtype=np.float32))
                for n in range(N):
                    if data[n, m] != 99:
                        u_n_T = u_T[n]
                        u_n = u_n_T.transpose()

                        temp1 = temp1 + u_n*u_n_T
                        temp2 = temp2 + data[n, m]*u_n

                I_k = np.matrix(np.identity(self.K, dtype=np.float32))
                temp3 = np.linalg.inv(temp1 + self.lambda_v*I_k)
                v_m = temp3*temp2
                v[:, m] = v_m

            for n in range(N):
                temp1 = np.matrix(np.zeros((self.K, self.K), dtype=np.float32))
                temp2 = np.matrix(np.zeros((self.K, 1), dtype=np.float32))
                for m in range(M):
                    if data[n, m] != 99:
                        v_m = v[:, m]
                        v_m_T = v_m.transpose()

                        temp1 = temp1 + v_m*v_m_T
                        temp2 = temp2 + data[n, m]*v_m

                I_k = np.matrix(np.identity(self.K, dtype=np.float32))
                temp3 = np.linalg.inv(temp1 + self.lambda_u * I_k)
                u_n = temp3*temp2
                u_T[n] = u_n.transpose()

            L_temp1 = L_temp2 = L_temp3 = 0

            for n in range(N):
                u_n_T = u_T[n]
                u_n = u_n_T.transpose()
                L_temp2 += self.lambda_u*(np.linalg.norm(u_n)**2)

            for m in range(M):
                v_m = v[:, m]
                L_temp3 += self.lambda_v*(np.linalg.norm(v_m)**2)

            for n in range(N):
                for m in range(M):
                    u_n_T = u_T[n]
                    v_m = v[:, m]
                    if data[n, m] != 99:
                        L_temp1 += (data[n, m] - u_n_T*v_m)**2

            L_reg = L_temp1 + L_temp2 + L_temp3
            if prev_L_reg == -100:
                prev_L_reg = L_reg
            else:
                if math.fabs(L_reg - prev_L_reg) <= 0.001:
                    break

    def test(self, test_data):
        res_matrix = np.matrix(self.u_T)*np.matrix(self.v)

        N = test_data.shape[0]
        M = test_data.shape[1]

        numerator = 0
        denominator = 0
        for n in range(N):
            for m in range(M):
                if test_data[n, m] != 99:
                    denominator += 1
                    numerator += (test_data[n, m] - res_matrix[n, m])**2

        rmse = math.sqrt(numerator/denominator)
        return rmse
