# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import cv2
from .utilities import sq
from OpenGL.GL import *
from OpenGL.GLUT import *
import warnings
import matplotlib.pyplot as plt


def normalize(v):
    return v / np.linalg.norm(v)


def intersect_line_line(p11, p12, p21, p22, internal=False):
    x1, y1 = p11
    x2, y2 = p12
    x3, y3 = p21
    x4, y4 = p22

    if ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) != 0:
        Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        if internal:
            if x1 != x2:
                lam = (Px - x2) / (x1 - x2)
            else:
                lam = (Py - y2) / (y1 - y2)
            if 0 <= lam <= 1:
                return [True, Px, Py]
            else:
                return [False]
        else:
            return [True, Px, Py]
    else:
        return [False]


def intersect_sphere_line(c_, r, o, l):
    c_ = np.asarray(c_).astype(np.float)
    o = np.asarray(o).astype(np.float)
    l = np.asarray(l).astype(np.float)
    temp = np.dot(l, o - c_)
    discriminant = temp ** 2 - np.linalg.norm(o - c_) ** 2 + r ** 2
    if discriminant >= 0.0:
        sqr = np.sqrt(discriminant)
        d1 = -temp + sqr
        d2 = -temp - sqr
        return [True, d1, d2]
    else:
        return [False]


def intersect_plane_line(p_plane, n_plane, p_line, l_line, radius=[]):
    p_plane = np.asarray(p_plane)
    n_plane = np.asarray(n_plane)
    p_line = np.asarray(p_line)
    l_line = np.asarray(l_line)

    if np.dot(n_plane, l_line) == 0 or np.dot(p_plane - p_line, n_plane) == 0:
        return [False]
    else:
        d = np.dot(p_plane - p_line, n_plane) / np.dot(l_line, n_plane)
        p_intersect = p_line + d * l_line
        if radius:
            if np.linalg.norm(p_plane - p_intersect) <= radius[0]:
                return [True, p_intersect[0], p_intersect[1], p_intersect[2]]
            else:
                return [False]
        else:
            return [True, p_intersect[0], p_intersect[1], p_intersect[2]]


def get_contour_pixels(img, offsets=[]):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = ((img < 0.01).astype(np.uint8) * 255)
        ret, thresh = cv2.threshold(img, 140, 255, 0)
        contours = cv2.findContours(thresh, 1, 1)
        c_ = contours[1][0]
        c_.shape = c_.shape[::2]
        if offsets:
            for i, p in enumerate(c_):
                c_[i] = c_[i] + np.asarray(offsets).astype(np.float64)
        return c_


def fit_ellipse(img, offsets=[]):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n,m = img.shape
    img = ((img < 0.01).astype(np.uint8) * 255)
    ret, thresh = cv2.threshold(img, 140, 255, 0)
    contours = cv2.findContours(thresh, 1, 1)
    c_ = contours[1][0]
    c_.shape = c_.shape[::2]
    c_ = [p for p in c_ if (2<p[0]<n-3 and 2<p[1]<m-3)]
    c_ = np.asarray(c_)
    if offsets:
        for i, p in enumerate(c_):
            c_[i] = c_[i] + np.asarray(offsets).astype(np.float64)
    ellipse = cv2.fitEllipse(c_)
    ellipse = ellipse_from_cv2_fit(ellipse)
    return ellipse, c_


def ellipse_from_cv2_fit(e):
    center = np.asarray((e[0][0], e[0][1]))  # THIS IS DUE TO OPENCV
    minor_radius = min(e[1][0], e[1][1]) / 2.
    major_radius = max(e[1][0], e[1][1]) / 2.
    angle = e[2] * np.pi / 180.

    return Ellipse(center, minor_radius, major_radius, angle)


class Conic(object):
    def __init__(self, *args):

        if len(args) == 1:
            ellipse = args[0]
            ax = np.cos(ellipse.angle)
            ay = np.sin(ellipse.angle)
            a2 = sq(ellipse.major_radius)
            b2 = sq(ellipse.minor_radius)

            self.A = ax * ax / a2 + ay * ay / b2
            self.B = 2. * ax * ay / a2 - 2. * ax * ay / b2
            self.C = ay * ay / a2 + ax * ax / b2
            self.D = (-2 * ax * ay * ellipse.center[1] - 2 * ax * ax * ellipse.center[0]) / a2 + (2 * ax * ay *
                                                                                                  ellipse.center[
                                                                                                      1] - 2 * ay * ay *
                                                                                                  ellipse.center[
                                                                                                      0]) / b2
            self.E = (-2 * ax * ay * ellipse.center[0] - 2 * ay * ay * ellipse.center[1]) / a2 + (2 * ax * ay *
                                                                                                  ellipse.center[
                                                                                                      0] - 2 * ax * ax *
                                                                                                  ellipse.center[
                                                                                                      1]) / b2
            self.F = (
                     2 * ax * ay * ellipse.center[0] * ellipse.center[1] + ax * ax * ellipse.center[0] * ellipse.center[
                         0] + ay * ay * ellipse.center[1] * ellipse.center[1]) / a2 + (
                                                                                      -2 * ax * ay * ellipse.center[0] *
                                                                                      ellipse.center[1] + ay * ay *
                                                                                      ellipse.center[
                                                                                          0] * ellipse.center[
                                                                                          0] + ax * ax *
                                                                                      ellipse.center[1] *
                                                                                      ellipse.center[
                                                                                          1]) / b2 - 1
        if len(args) == 6:
            self.A, self.B, self.C, self.D, self.E, self.F = args

    def discriminant(self):
        return self.B ** 2 - 4 * self.A * self.C


class Conidcoid(object):
    def __init__(self, conic, vertex):
        alpha = vertex[0]
        beta = vertex[1]
        gamma = vertex[2]
        self.A = sq(gamma) * conic.A
        self.B = sq(gamma) * conic.C
        self.C = conic.A * sq(alpha) + conic.B * alpha * beta + conic.C * sq(
            beta) + conic.D * alpha + conic.E * beta + conic.F
        self.F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2)
        self.G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2)
        self.H = sq(gamma) * conic.B / 2
        self.U = sq(gamma) * conic.D / 2
        self.V = sq(gamma) * conic.E / 2
        self.W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F)
        self.D = sq(gamma) * conic.F


def draw_cone_from_ellipse_gl(ellipse, camera):
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glColor4f(1, 1, 0, 1.0)

    glPushMatrix()
    glTranslate(0, 0, -camera.focal_length)
    glScale(7, 7, 7)
    phis = np.linspace(0, 2 * np.pi, 20)
    for i in range(19):
        glBegin(GL_TRIANGLES)
        p1 = parametrize_ellipse(phis[i], ellipse)
        p2 = parametrize_ellipse(phis[i + 1], ellipse)
        glVertex3f(p1[0], p1[1], camera.focal_length)
        glVertex3f(p2[0], p2[1], camera.focal_length)
        glVertex3f(0, 0, 0)
        glEnd()
    glPopMatrix()


class Circle(object):
    def __init__(self, center, normal, radius):
        self.center = center
        self.normal = normal
        self.radius = radius

        self.tangent_1 = np.cross(self.normal, [0, 0.1, 1.1])
        self.tangent_2 = np.cross(self.normal, self.tangent_1)

        self.tangent_1 /= np.linalg.norm(self.tangent_1)
        self.tangent_2 /= np.linalg.norm(self.tangent_2)

    def print(self):
        print(
            "Center: {:f}, {:f}, {:f} - Radius: {:f} - Normal: {:f}, {:f}, {:f}".format(self.center[0], self.center[1],
                                                                                        self.center[2],
                                                                                        self.radius,
                                                                                        self.normal[0], self.normal[1],
                                                                                        self.normal[2]))


class Ellipse(object):
    def __init__(self, center, minor_radius, major_radius, angle):
        self.center = center
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.angle = angle

    def print(self):
        print(self.center, self.minor_radius, self.major_radius, self.angle)

    def area(self):
        return np.pi * self.minor_radius * self.major_radius


def unproject_ellipse(ellipse, circle_radius, focal_length):
    try:
        conic = Conic(ellipse)
        pupil_cone = Conidcoid(conic, [0, 0, -focal_length])

        a = pupil_cone.A
        b = pupil_cone.B
        c = pupil_cone.C
        f = pupil_cone.F
        g = pupil_cone.G
        h = pupil_cone.H
        u = pupil_cone.U
        v = pupil_cone.V
        w = pupil_cone.W

        p = np.zeros(4)

        p[0] = 1
        p[1] = -(a + b + c)
        p[2] = (b * c + c * a + a * b - f * f - g * g - h * h)
        p[3] = -(a * b * c + 2 * f * g * h - a * f * f - b * g * g - c * h * h)
        lambda_ = np.roots(p)

        n = np.sqrt((lambda_[1] - lambda_[2]) / (lambda_[0] - lambda_[2]))
        m = 0.0
        l = np.sqrt((lambda_[0] - lambda_[1]) / (lambda_[0] - lambda_[2]))

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                t1 = (b - lambda_) * g - f * h
                t2 = (a - lambda_) * f - g * h
                t3 = -(a - lambda_) * (t1 / t2) / g - h / g
            except Warning as e:
                # This happens when t2 is [0,0,0]
                print(e)
                return []

        mi = 1 / np.sqrt(1 + (t1 / t2) ** 2 + t3 ** 2)
        li = (t1 / t2) * mi
        ni = t3 * mi

        if (np.dot(np.cross(li, mi), ni) < 0):
            li = -li
            mi = -mi
            ni = -ni

        T1 = np.asarray([li, mi, ni])

        T2 = - (u * li + v * mi + w * ni) / lambda_

        solution_circles = []

        for l in [l, -l]:

            if (l == 0):
                assert (n == 1)
                T3 = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            else:
                T3 = np.asarray([[0, -n * np.sign(l), l], [np.sign(l), 0, 0], [0, np.abs(l), n]])

            A = np.dot(lambda_, T3[:, 0] ** 2)
            B = np.dot(lambda_, T3[:, 0] * T3[:, 2])
            C = np.dot(lambda_, T3[:, 1] * T3[:, 2])
            D = np.dot(lambda_, T3[:, 2] ** 2)

            center_in_Xprime = np.zeros(3)
            center_in_Xprime[2] = A * circle_radius / np.sqrt(sq(B) + sq(C) - A * D)
            center_in_Xprime[0] = -B / A * center_in_Xprime[2]
            center_in_Xprime[1] = -C / A * center_in_Xprime[2]

            T0 = [0, 0, focal_length]

            center = np.dot(T1, np.dot(T3, center_in_Xprime) + T2) + T0
            if center[2] < 0:
                center_in_Xprime = -center_in_Xprime
                center = np.dot(T1, np.dot(T3, center_in_Xprime) + T2) + T0
            center += np.array([0, 0, -focal_length])  # SHIFT INTO IMAGE COORDINATE SYSTEM

            gaze = np.dot(T1, np.asarray([l, m, n]))
            if np.dot(gaze, center) > 0:
                gaze = -gaze
            gaze = normalize(gaze)

            solution_circles.append(Circle(center, gaze, circle_radius))

        return solution_circles
    except:
        return False


def parametrize_ellipse(phi, ellipse):
    M = np.array([[np.cos(ellipse.angle), -np.sin(ellipse.angle)], [np.sin(ellipse.angle), np.cos(ellipse.angle)]])
    e1 = np.dot(M, np.array([1., 0.]))
    e2 = np.dot(M, np.array([0., 1.]))

    return ellipse.center + ellipse.major_radius * np.cos(phi) * e1 + ellipse.minor_radius * np.sin(phi) * e2


def ellipse_points(ellipse):

    phis = np.linspace(0,2*np.pi,100)
    x = [parametrize_ellipse(phi,ellipse)[0] for phi in phis]
    y = [parametrize_ellipse(phi,ellipse)[1] for phi in phis]
    return x,y


def draw_ellipse_gl(ellipse, z, c=(1, 0, 0), lw=1):
    glColor3f(c[0], c[1], c[2])
    glLineWidth(lw)

    glPushMatrix()
    glBegin(GL_LINE_STRIP)
    for phi in np.linspace(0, 2 * np.pi, 50):
        p = parametrize_ellipse(phi, ellipse)
        glVertex3f(p[0], p[1], z)
    glEnd()
    glPopMatrix()


def draw_circle_gl(circle, radius, color=(0, 1, 0), alpha=1.0, lw=1.0):
    glLineWidth(lw)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glColor4f(color[0], color[1], color[2], alpha)
    glPushMatrix()
    phis = np.linspace(0, 2 * np.pi, 15)
    for i in range(14):
        x1 = radius * np.cos(phis[i]) * circle.tangent_1
        y1 = radius * np.sin(phis[i]) * circle.tangent_2
        x2 = radius * np.cos(phis[i + 1]) * circle.tangent_1
        y2 = radius * np.sin(phis[i + 1]) * circle.tangent_2
        glBegin(GL_TRIANGLES)
        glVertex3f(circle.center[0], circle.center[1], circle.center[2])
        glVertex3f(circle.center[0] + x1[0] + y1[0], circle.center[1] + x1[1] + y1[1], circle.center[2] + x1[2] + y1[2])
        glVertex3f(circle.center[0] + x2[0] + y2[0], circle.center[1] + x2[1] + y2[1], circle.center[2] + x2[2] + y2[2])
        glEnd()
    glPopMatrix()


def project_vector_into_image_plane(x, normal, normalize_=True):
    x_proj = x - np.dot(normal, x) * normal / np.linalg.norm(normal)
    if normalize_ == True:
        x_proj = normalize(x_proj)
    return x_proj


def project_point_into_image_plane(x, focal_length):
    x_temp = x - np.asarray([0, 0, -focal_length])
    scale = focal_length / x_temp[2]
    x_projected = scale * x_temp
    x_projected[2] = 0
    return x_projected


def draw_line_gl(p, l, extent=10):
    glPushMatrix()
    glColor3f(1, 1, 0)
    glBegin(GL_LINE_STRIP)
    glVertex3f(p[0] - extent * l[0], p[1] - extent * l[1], p[2] - extent * l[2])
    glVertex3f(p[0], p[1], p[2])
    glVertex3f(p[0] + extent * l[0], p[1] + extent * l[1], p[2] + extent * l[2])
    glEnd()
    glPopMatrix()


def unproject_point(p, z=0, focal_length=10.0):
    lam_ = (z + focal_length) / focal_length

    p0_new = lam_ * p[0]
    p1_new = lam_ * p[1]
    p2_new = z

    return np.array([p0_new, p1_new, p2_new], dtype=np.float32)


def draw_coordinate_system_gl():
    glColor3f(0, 1, 0)
    glBegin(GL_LINE_STRIP)
    glVertex3f(0, 0, 0)
    glVertex3f(300, 0, 0)  # X-axis = green
    glEnd()

    glColor3f(1, 0, 0)
    glBegin(GL_LINE_STRIP)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 300, 0)  # Y-axis = red
    glEnd()

    glColor3f(0, 0, 1)
    glBegin(GL_LINE_STRIP)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 300)  # Z-axis = blue
    glEnd()


def mark_point_gl(p, radius=0.05, lw=1.0, c_ = [1,0,0]):
    glPushMatrix()
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glLineWidth(lw)
    glColor(c_[0], c_[1], c_[2])
    glTranslate(p[0], p[1], p[2])
    glutWireSphere(radius, 20, 20)
    glPopMatrix()


def spherical_coordinates(p, c):
    v = p - c
    r = np.linalg.norm(v)
    theta__ = np.arccos(v[2] / r)
    phi = np.arctan2(v[1], v[0])

    return phi, theta__


def spherical_coordinates_cpp(p, c=np.array([0,0,0])):
    v = p - c
    r = np.linalg.norm(v)
    theta = np.arccos(v[1] / r)
    phi = np.arctan2(v[2], v[0])

    return theta, phi


def from_spherical_cpp(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.sin(phi)

    return np.asarray([x, y, z])


def from_spherical(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.asarray([x, y, z])


def approximate_distance_points_to_ellipse(ps, ellipse, plot=False, ax=None):

    M = np.array([[np.cos(ellipse.angle), -np.sin(ellipse.angle)], [np.sin(ellipse.angle), np.cos(ellipse.angle)]])
    v1 = 1. / ellipse.major_radius * np.dot(M, np.asarray([1, 0]))
    v2 = 1. / ellipse.minor_radius * np.dot(M, np.asarray([0, 1]))
    R = [[v1[0], v1[1]], [v2[0], v2[1]]]

    distances = []

    if plot:
        x, y = ellipse_points(ellipse)
        plt.plot(x, y, "r")
        for p in ps:
            ax.plot(p[0], p[1], "bo", alpha=0.5)

    for p in ps:

        pprime = np.dot(R, p - np.asarray(ellipse.center))
        # if plot:
        #     ax.plot(pprime[0], pprime[1], "ro")
        d_signed = np.linalg.norm(pprime) - 1
        d_scaled = d_signed * ellipse.major_radius
        distances.append(d_scaled)

    return distances


def rotate_v1_on_v2(v1, v2):

    if not np.linalg.norm(v1-v2) == 0.0:
        u = np.cross(v1, v2)
        s = np.linalg.norm(u)
        c = np.dot(v1, v2)

        I = np.eye(3)
        ux = np.asarray([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

        R = I + ux + np.dot(ux, ux) * (1 - c) / s ** 2

    else:
        R = np.eye(3)

    return R


if __name__=="__main__":

    from projections import *
    from eye import Eye
    eye = Eye()
    R = rotate_v1_on_v2([0,0,-1],normalize([0.5, 0.5, -0.5]))
    eye.rotate_eye(R, axis='R')
    circle = Circle(eye.pupil_center, eye.pupil_normal, eye.pupil_radius)
    #ellipse = project_circle_into_image_plane_refraction(circle, eye.eyeball_center, n=1.3375, focal_length=10, resolution_=100)
    #ellipse.print()