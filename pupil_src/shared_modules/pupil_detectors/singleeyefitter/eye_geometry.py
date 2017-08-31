from OpenGL.GL import *
import numpy as np

from .utilities import rmatx, rmaty, rmatz, rotate
from .geometry import normalize, Circle, rotate_v1_on_v2

class Eye(object):
    def __init__(self, pupil_radius=3.0, n=1.0, de=50.0, re=12.0, rc=7.3, ri=5.9):

        self.d_image_eyeball = de  # mm

        self.d_eyeball_pupil = np.sqrt(re**2-ri**2) #dp  # mm
        self.h = np.sqrt(rc**2-ri**2) # mm
        self.d_eyeball_cornea = self.d_eyeball_pupil-self.h  # mm

        # CORNEA
        self.cornea_center = np.asarray([0, 0, self.d_image_eyeball - self.d_eyeball_cornea])
        self.cornea_radius = rc  # mm
        self.cornea_alpha = np.arccos(4. / self.cornea_radius)/1.1

        # PUPIL
        self.pupil_radius = pupil_radius # mm
        self.pupil_center = [0., 0., self.d_image_eyeball - self.d_eyeball_pupil]
        self.pupil_normal = [0., 0., -1.]

        # IRIS
        self.iris_radius = ri #np.sqrt(self.cornea_radius ** 2 - (self.d_eyeball_pupil - self.d_eyeball_cornea) ** 2)  # L
        self.iris_center = [0., 0., self.d_image_eyeball - self.d_eyeball_pupil]
        self.iris_normal = [0., 0., -1.]

        # EYEBALL
        self.eyeball_center = np.asarray([0., 0., self.d_image_eyeball])
        self.eyeball_radius = re #np.sqrt(self.iris_radius ** 2 + self.d_eyeball_pupil ** 2)
        self.eyeball_alpha = np.arccos(self.d_eyeball_pupil / self.eyeball_radius)

        # GAZE VECTOR
        self.gaze_vector = normalize(self.pupil_center-self.eyeball_center)
        self.phi_x = 0.
        self.phi_y = 0.
        self.phi_z = 0.

        # PHYSICAL CONSTANTS
        self.n = n  # REFRACTION INDEX OF CORNEA # 1 or 1.3375

        self.R = np.identity(3)

        self.set_up_vertices()

    def update_geometry(self):

        self.d_eyeball_pupil = np.sqrt(self.eyeball_radius ** 2 - self.iris_radius ** 2)  # dp  # mm
        self.h = np.sqrt(self.cornea_radius ** 2 - self.iris_radius ** 2)  # mm
        self.d_eyeball_cornea = self.d_eyeball_pupil - self.h  # mm

        # CORNEA
        self.cornea_center = np.asarray([0, 0, self.d_image_eyeball - self.d_eyeball_cornea])
        self.cornea_alpha = np.arccos(4. / self.cornea_radius)/1.1

        # PUPIL
        self.pupil_center = [0., 0., self.d_image_eyeball - self.d_eyeball_pupil]

        # IRIS
        self.iris_center = [0., 0., self.d_image_eyeball - self.d_eyeball_pupil]

        # EYEBALL
        self.eyeball_center = np.asarray([0., 0., self.d_image_eyeball])
        self.eyeball_alpha = np.arccos(self.d_eyeball_pupil / self.eyeball_radius)

        # CURRENT ROTATION MATRIX
        R = self.R
        self.rotate_eye(R, axis='R')

        # NEEDED FOR NEW INTEGRATION OF ROTATIONS
        self.R = np.identity(3)

        # VERTEX LISTS FOR DRAWING
        self.set_up_vertices()

    def update_gaze_vector(self):

        self.gaze_vector = normalize(self.pupil_center-self.eyeball_center)

    def rotate_eye(self, theta, axis='x'):

        if axis == 'x':
            R = rmatx(theta)
            self.phi_x += theta
        if axis == 'y':
            R = rmaty(theta)
            self.phi_y += theta
        if axis == 'z':
            R = rmatz(theta)
            self.phi_z += theta
        if axis=='R':
            R = theta

        self.cornea_center = rotate(R, self.cornea_center, self.eyeball_center)  # ROTATION AROUND EYEBALL CENTER
        self.pupil_center = rotate(R, self.pupil_center, self.eyeball_center)  # ROTATION AROUND EYEBALL CENTER
        self.pupil_normal = rotate(R, self.pupil_normal)
        self.iris_center = rotate(R, self.iris_center, self.eyeball_center)  # ROTATION AROUND EYEBALL CENTER
        self.iris_normal = rotate(R, self.iris_normal)
        self.gaze_vector = rotate(R, self.gaze_vector)

        self.R = R@self.R
        self.update_gaze_vector()

    def translate_eye(self, t):

        t = np.asarray(t)
        self.cornea_center += t
        self.pupil_center += t
        self.iris_center += t
        self.eyeball_center += t

    def move_to_point(self, p):
        p_start = self.eyeball_center
        p_end = np.asarray(p)
        self.translate_eye(p_end-p_start)

    def rotate_to_gaze_vector(self, gv_):
        rot = rotate_v1_on_v2(self.gaze_vector, normalize(gv_))
        self.rotate_eye(rot, axis='R')

    def print(self):

        print(
            "Eyeball center: {}\nCornea center: {}\n Pupil center: {}\n Pupil normal: {}\n Iris center: {}\n Iris normal {}\n".format(
                self.eyeball_center,
                self.cornea_center,
                self.pupil_center,
                self.pupil_normal,
                self.iris_center,
                self.iris_normal))

    def set_up_vertices(self):

        # EYEBALL
        self.central_ring_eyeball = [[self.eyeball_radius * np.sin(phi), 0, self.eyeball_radius * np.cos(phi)]
                                     for phi in np.linspace(-np.pi + self.eyeball_alpha, np.pi - self.eyeball_alpha, 30)]
        self.rings_eyeball = [self.central_ring_eyeball]
        for phi in np.linspace(0, np.pi, 20):
            central_ring_rotated = [rotate(rmatz(phi), v) for v in self.central_ring_eyeball]
            self.rings_eyeball.append(central_ring_rotated)

        # IRIS
        angles = [phi for phi in np.linspace(0, 2 * np.pi, 10)]
        self.iris_quads = []
        for i in range(len(angles) - 1):
            self.iris_quads.append([np.array((np.cos(angles[i]), np.sin(angles[i]), 0)),
                                    np.array((np.cos(angles[i + 1]), np.sin(angles[i + 1]), 0)),
                                    np.array((np.cos(angles[i + 1]), np.sin(angles[i + 1]), 0)),
                                    np.array((np.cos(angles[i]), np.sin(angles[i]), 0))])

        # CORNEA
        self.central_ring_cornea = [[self.cornea_radius * np.sin(phi), 0, self.cornea_radius * np.cos(phi)]
                                    for phi in np.linspace(-np.pi + self.cornea_alpha, -np.pi, 10)] \
                                 + [[self.cornea_radius * np.sin(phi), 0, self.cornea_radius * np.cos(phi)]
                                    for phi in np.linspace(np.pi, np.pi - self.cornea_alpha, 10)]
        self.rings_cornea = [self.central_ring_cornea]
        for phi in np.linspace(0, np.pi, 10):
            central_ring_rotated = [rotate(rmatz(phi), v) for v in self.central_ring_cornea]
            self.rings_cornea.append(central_ring_rotated)

    def draw_model(self, eyeball=True, iris=True, cornea=True, ic=[46/255., 220/255., 255./255.]):

        glPushMatrix()

        a = (GLfloat * 16)()
        glGetFloatv(GL_MODELVIEW_MATRIX, a)
        a = np.reshape(a, (4, 4))
        R = np.identity(4)
        R[:3,:3] = rotate_v1_on_v2([0, 0, -1], self.gaze_vector)
        a = a@R
        a[3, 0:3] = -self.eyeball_center
        a[3, 0] *= -1.0
        a[3, 1] *= -1.0

        glLoadMatrixf(a)

        glPushMatrix()

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # DRAW EYEBALL
        if eyeball:

            glColor4f(0.6, 0.6, 1.0, 0.7)
            glLineWidth(2.0)

            glPushMatrix()
            for i in range(len(self.rings_eyeball) - 1):
                for j in range(len(self.rings_eyeball[i]) - 1):
                    glBegin(GL_QUADS)
                    glVertex3f(self.rings_eyeball[i][j][0], self.rings_eyeball[i][j][1], self.rings_eyeball[i][j][2])
                    glVertex3f(self.rings_eyeball[i][j + 1][0], self.rings_eyeball[i][j + 1][1], self.rings_eyeball[i][j + 1][2])
                    glVertex3f(self.rings_eyeball[i + 1][j + 1][0], self.rings_eyeball[i + 1][j + 1][1], self.rings_eyeball[i + 1][j + 1][2])
                    glVertex3f(self.rings_eyeball[i + 1][j][0], self.rings_eyeball[i + 1][j][1], self.rings_eyeball[i + 1][j][2])
                    glEnd()
            glPopMatrix()

        # DRAW IRIS
        if iris:

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glColor4f(ic[0], ic[1], ic[2], 0.3)

            glPushMatrix()
            glTranslate(0, 0, -self.d_eyeball_pupil)
            for quad in self.iris_quads:
                glBegin(GL_QUADS)
                glVertex3f(*(quad[0]*self.pupil_radius))
                glVertex3f(*(quad[1]*self.pupil_radius))
                glVertex3f(*(quad[2]*self.iris_radius))
                glVertex3f(*(quad[3]*self.iris_radius))
                glEnd()
            glPopMatrix()

        # DRAW CORNEA
        if cornea:

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor4f(1, 1, 1, 0.3)
            glLineWidth(1.0)

            glPushMatrix()
            glTranslate(0, 0, -self.d_eyeball_cornea)
            for i in range(len(self.rings_cornea) - 1):
                for j in range(len(self.rings_cornea[i]) - 1):
                    glBegin(GL_QUADS)
                    glVertex3f(self.rings_cornea[i][j][0], self.rings_cornea[i][j][1], self.rings_cornea[i][j][2])
                    glVertex3f(self.rings_cornea[i][j + 1][0], self.rings_cornea[i][j + 1][1], self.rings_cornea[i][j + 1][2])
                    glVertex3f(self.rings_cornea[i + 1][j + 1][0], self.rings_cornea[i + 1][j + 1][1], self.rings_cornea[i + 1][j + 1][2])
                    glVertex3f(self.rings_cornea[i + 1][j][0], self.rings_cornea[i + 1][j][1], self.rings_cornea[i + 1][j][2])
                    glEnd()
            glPopMatrix()


        glPopMatrix()

        glPopMatrix()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

def get_pupil_from_parameters(phi, theta, r):

    eye = Eye()
    eye.rotate_eye(phi, axis='y')
    eye.rotate_eye(theta, axis='z')
    pupil = Circle(eye.pupil_center, eye.pupil_normal, r)
    return pupil

if __name__ == "__main__":

    # eye = Eye()
    # v = normalize([0.5, 0.5, 0.5])
    # R = rotate_v1_on_v2([0, 0, -1], v)
    # eye.rotate_eye(R, axis='R')
    # eye.translate_eye([0,0,1])
    # print(eye.gaze_vector)

    eye = Eye()
    eye.rotate_eye(0, axis='y')
    eye.rotate_eye(0, axis='z')
    print(eye.gaze_vector)
