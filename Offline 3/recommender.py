import numpy as np
import random as rand
import math
import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse


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
                temp.append(rand.uniform(-1, 1))
            u_T.append(temp)

        u_T = np.matrix(u_T).astype(np.float32)

        for i in range(M):
            temp = []
            for k in range(self.K):
                temp.append(rand.uniform(-1, 1))
            v.append(temp)

        v = np.matrix(v).transpose().astype(np.float32)

        prev_L_reg = -100

        iteration = 1

        temp1_ = T.matrix('temp1_')
        temp2_ = T.matrix('temp2_')
        vec_T_ = T.matrix('u_n_T_')
        vec_ = T.matrix('u_n_')
        d_n_m = T.scalar('d_n_m')
        res1 = temp1_ + T.dot(vec_, vec_T_)
        res2 = temp2_ + d_n_m*vec_
        f1 = theano.function([temp1_, vec_, vec_T_], res1)
        f2 = theano.function([temp2_, d_n_m, vec_], res2)

        lambda_ = T.scalar('lambda_')
        I_k_ = T.matrix('I_k_')
        res3 = matrix_inverse(temp1_ + lambda_*I_k_)
        f3 = theano.function([temp1_, lambda_, I_k_], res3)

        temp3_ = T.matrix('temp3_')
        res4 = T.dot(temp3_, temp2_)
        f4 = theano.function([temp3_, temp2_], res4)

        while True:
            print("Iteration = " + str(iteration))

            print("Updating Vm vectors")

            for m in range(M):
                temp1 = np.matrix(np.zeros((self.K, self.K), dtype=np.float32))
                temp2 = np.matrix(np.zeros((self.K, 1), dtype=np.float32))
                for n in range(N):
                    if data[n, m] != 99:
                        u_n_T = u_T[n]
                        u_n = u_n_T.transpose()

                        temp1 = f1(temp1, u_n, u_n_T)
                        temp2 = f2(temp2, data[n, m], u_n)

                I_k = np.matrix(np.identity(self.K, dtype=np.float32))
                temp3 = f3(temp1, self.lambda_v, I_k)
                v_m = f4(temp3, temp2)
                for k in range(self.K):
                    v[k, m] = v_m[k, 0]
            self.v = v
            print("Updating Un_T vectors")
            for n in range(N):
                temp1 = np.matrix(np.zeros((self.K, self.K), dtype=np.float32))
                temp2 = np.matrix(np.zeros((self.K, 1), dtype=np.float32))
                for m in range(M):
                    if data[n, m] != 99:
                        v_m = v[:, m]
                        v_m_T = v_m.transpose()

                        temp1 = f1(temp1, v_m, v_m_T)
                        temp2 = f2(temp2, data[n, m], v_m)

                I_k = np.matrix(np.identity(self.K, dtype=np.float32))
                temp3 = f3(temp1, self.lambda_u, I_k)
                u_n = f4(temp3, temp2)
                u_n_T = u_n.transpose()
                u_T[n] = u_n_T
            self.u_T = u_T
            L_temp1 = L_temp2 = L_temp3 = 0

            print("Calculating empirical risk")
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
                        L_temp1 += (data[n, m] - f4(u_n_T, v_m))**2

            L_reg = L_temp1 + L_temp2 + L_temp3
            print("Empirical Risk = " + str(L_reg[0, 0]))
            if prev_L_reg != -100:
                if math.fabs(L_reg[0, 0] - prev_L_reg) <= self.threshold:
                    break
            prev_L_reg = L_reg[0, 0]
            iteration += 1

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

        RMSE = math.sqrt(numerator/denominator)
        return RMSE
