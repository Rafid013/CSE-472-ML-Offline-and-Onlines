import math


def normalize(w):
    temp = w[:]
    s = float(sum(temp))

    for i in range(0, len(temp)):
        temp[i] /= s
    return temp


class AdaBoost:
    def __init__(self, examples, weak_learner, K):
        self.examples = examples
        self.weak_learner = weak_learner
        self.K = K
        self.N = examples.shape[0]
        self.w = []

        for i in range(0, self.N):
            self.w.append(1/float(self.N))

        self.h = []
        self.z = []

    def train(self):
        examples = self.examples
        weak_learner = self.weak_learner
        K = self.K
        w = self.w
        h = self.h
        z = self.z
        N = self.N

        for k in range(0, K):
            data = examples.sample(n=N, replace=True, weights=w)

            labels = data.iloc[:, data.shape[1] - 1]
            attributes = data.iloc[:, :data.shape[1] - 1]

            learner = weak_learner(1)
            learner.train(labels, attributes)
            h.append(learner)

            error = 0

            for j in range(0, N):
                xj = examples.iloc[j, :examples.shape[1] - 1]
                yj = examples.iloc[j, examples.shape[1] - 1]
                res = h[k].decide(xj.tolist())

                if res != yj:
                    error += w[j]
            if error > 0.5:
                continue
            for j in range(0, N):
                xj = examples.iloc[j, :examples.shape[1] - 1]
                yj = examples.iloc[j, examples.shape[1] - 1]
                res = h[k].decide(xj.tolist())

                if res == yj:
                    w[j] *= error/(1 - error)

            w = normalize(w)
            z.append(math.log((1 - error)/error, 2))

        self.h = h
        self.z = z

    def decide(self, attributes):
        h = self.h
        z = self.z
        s = 0
        for k in range(0, self.K):
            res = h[k].decide(attributes)
            if res == 0:
                res = -1
            s += (res*z[k])
        if s >= 0:
            return 1
        return 0
