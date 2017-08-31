import numpy as np
from numpy import cos, sin, arccos, arcsin, pi
from numpy.linalg import norm
from scipy.optimize import brentq

dc = 6.0
dp = 10.0
de = 50.0
f = 10.0
rc = 9.5
rp = 3.0
ri = np.sqrt(rc ** 2 - (dp - dc) ** 2)
re = np.sqrt(ri ** 2 + dp ** 2)

phi_max = arcsin(ri / rc)

Q0 = np.asarray([dc, 0])
P0 = np.asarray([dp, 0])
C = np.asarray([de + f, 0])

n = 1.3375

def D(th):
    return np.asarray([[cos(th), -sin(th)], [sin(th), cos(th)]])


def Q(th):
    return np.dot(D(th), Q0)


def P(th):
    return np.dot(D(th), P0)


def PQ(th):
    return P(th) - Q(th)


def F(phi, th):
    return np.dot(D(th), Q0 + rc * np.asarray([cos(phi), sin(phi)]))


def CF(phi, th):
    return F(phi, th) - C


def CQ(th):
    return Q(th) - C


def tangent(phi, th):
    t_ = np.asarray([-sin(phi), cos(phi)])
    return np.dot(D(th), t_)


def alpha(phi, th):
    temp = norm(CF(phi, th)) ** 2 + rc ** 2 - norm(CQ(th)) ** 2
    temp /= (2.0 * norm(CF(phi, th)) * rc)
    if np.dot(tangent(phi, th), CQ(th)) > 0:
        return pi - arccos(temp)
    else:
        return -(pi - arccos(temp))


def beta(phi, th):
    temp = (dp - dc) ** 2 - norm(P(th) - F(phi, th)) ** 2 - rc ** 2
    temp /= 2 * norm(P(th) - F(phi, th)) * rc
    if np.dot(tangent(phi, th), PQ(th)) > 0:
        return arccos(temp)
    else:
        return -arccos(temp)


def intersection(th, n=1.3375):
    return brentq(lambda x: sin(alpha(x, th)) - n*sin(beta(x, th)), -phi_max, phi_max)


def get_gaze_angle_for_shifted_sphere(th, Delta=10.0, D=60.0, r=10.0, n=1.3375):

    phi = intersection(th, n=n)
    gamma = arcsin(F(phi, th)[1]/norm(CF(phi, th)))

    d1 = (D - Delta) * np.cos(gamma) + np.sqrt(((D - Delta) ** 2) * (np.cos(gamma) ** 2) - (D - Delta) ** 2 + r ** 2)
    d2 = (D - Delta) * np.cos(gamma) - np.sqrt(((D - Delta) ** 2) * (np.cos(gamma) ** 2) - (D - Delta) ** 2 + r ** 2)
    d = min(d1, d2)  # CHOOSING THE INTERSECTION CLOSER TO THE CAMERA

    beta = np.arccos((r ** 2 + (D - Delta) ** 2 - d ** 2) / (2. * r * (D - Delta)))

    return beta


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    thetas = np.linspace(0, pi / 4, 1000)

    ints = [intersection(th) for th in thetas]
    gammas = [arcsin(F(ints[i], thetas[i])[1]/norm(CF(ints[i], thetas[i]))) for i in range(len(thetas))]
    ga = np.asarray([gaze_angle(th) for th in thetas])
    # plt.plot(thetas*180/np.pi, (thetas-ga)*180/np.pi)
    # plt.show()

    test = np.zeros((1000,2))
    test[:, 0] = thetas*180/pi
    test[:, 1] = (thetas-ga)*180/pi
    np.save("/cluster/Kai/refraction/results/ga.npy",test)