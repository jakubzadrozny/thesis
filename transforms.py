import numpy as np

class RandomRotation(object):
    @staticmethod
    def gen_x_matrix(a):
        return np.array([
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ])

    @staticmethod
    def gen_y_matrix(a):
        return np.array([
            [np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0, np.cos(a)]
        ])

    @staticmethod
    def gen_z_matrix(a):
        return np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def get_matrix(a, b, c):
        X = RandomRotation.gen_x_matrix(a)
        Y = RandomRotation.gen_y_matrix(b)
        Z = RandomRotation.gen_z_matrix(c)
        return np.dot(X, np.dot(Y, Z))

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, sample):
        a, b, c = np.astype(self.scale * np.pi * np.abs(np.random.normal(size=3))), dtype='float')
        R = RandomRotation.get_matrix(a, b, c)
        res = np.dot(R, sample)
        return res
