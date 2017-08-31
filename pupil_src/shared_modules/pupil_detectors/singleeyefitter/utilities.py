import numpy as np


def rmatx(theta):
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])


def rmaty(theta):
    return np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])


def rmatz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0], [0.0, 0.0, 1]])


def rotate(R, v, offset=(0, 0, 0)):
    v = np.asarray(v)
    offset = np.asarray(offset)
    return np.dot(R, v - offset) + offset


def sq(x):
    return x ** 2


def vector_orthgonal_to_two_vectors(v1, v2):
    return np.cross(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2))


def correction_matrix(v1, v2, theta_):
    axis = vector_orthgonal_to_two_vectors(v1, v2)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta_ / 2.0)
    b, c, d = -axis * np.sin(theta_ / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])





