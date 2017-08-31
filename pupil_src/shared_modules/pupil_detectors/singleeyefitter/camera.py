import sys
# sys.path.append("/home/kd/pupil/pupil_src/capture")
sys.path.append("/home/kd/git/pupil/pupil_src/shared_modules")
from .geometry import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import PIL.Image
import cv2
try:
    import pupil_detectors
    from video_capture.fake_backend import Frame
    from methods import Roi
except:
    pass
import matplotlib.pyplot as plt
from .eye_geometry import *

texture = 0

class Camera(object):
    def __init__(self, X=5.0, Y=5.0, pixels_x=800, pixels_y=800, focal_length=10):

        self.X = X
        self.Y = Y
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.dx = X / pixels_x
        self.dy = Y / pixels_y
        self.focal_length = focal_length
        self.ellipse = None
        self.ellipse_list = []
        self.ellipse_contour_list = []
        self.circles = None
        self.circles_list = []

        self.video_capture = None
        self.counter = 0
        self.image = None
        self.frame = None


        try:
            self.pupil_detector = pupil_detectors.Detector_2D()
        except:
            pass

    def reset(self):

        self.ellipse = None
        self.ellipse_list = []
        self.ellipse_contour_list = []
        self.circles = None
        self.circles_list = []
        self.counter = 0


    def reset_image(self):
        self.image = np.ones((self.pixels_y, self.pixels_x, 3), dtype=np.float32)

    def load_image(self, file_):

        with np.load(file_) as data:
            self.image = data['arr_0']

        phi = float(file_.split("/")[-1].split("_")[2])
        theta = float(file_.split("/")[-1].split("_")[3])
        r = float(file_.split("/")[-1].split("_")[4])

        return phi, theta, r

    def make_image_iterative(self, eye, initial=10, steps=5, factor=2):

        self.reset_image()

        origin = np.asarray((0, 0, -self.focal_length))

        for step in range(steps+1):
            current_px = initial*factor**step
            if step==0:
                toggle = np.ones((current_px, current_px))
            current_px_size = self.X/current_px
            current_image = np.zeros((current_px*factor, current_px*factor, 3), dtype=np.float32)
            N = int(factor**(steps-step))
            for i in range(current_px):
                for j in range(current_px):
                    if toggle[j, i] == 1:
                        x = -self.X / 2 + current_px_size / 2. + i * current_px_size
                        y = -self.Y / 2 + current_px_size / 2. + j * current_px_size
                        co = self.detect_pixel(eye, x, y, origin)
                        current_image[j*factor:(j+1)*factor, i*factor:(i+1)*factor] = co
                        self.image[j*N:(j+1)*N, i*N:(i+1)*N] = co
            toggle = np.zeros((current_px * factor, current_px * factor))
            c_ = get_contour_pixels(current_image)
            for p in c_:
                toggle[p[1]-2*factor:p[1]+2*factor,p[0]-2*factor:p[0]+2*factor] = 1

    def make_image(self, eye):

        self.reset_image()
        origin = np.asarray((0, 0, -self.focal_length))
        for i in range(self.pixels_x):
            for j in range(self.pixels_y):
                x = -self.X / 2 + self.dx / 2. + i * self.dx
                y = -self.Y / 2 + self.dy / 2. + j * self.dy
                co = self.detect_pixel(eye, x, y, origin)
                self.image[j][i] = co

    def detect_pixel(self, eye, x, y, origin):
        z = self.focal_length
        v = np.asarray([x, y, z]).astype(np.float)
        v = normalize(v)
        intersect_cornea = (intersect_sphere_line(eye.cornea_center, eye.cornea_radius, origin, v))
        intersect_eyeball = (intersect_sphere_line(eye.eyeball_center, eye.eyeball_radius, origin, v))
        if intersect_cornea[0] and intersect_eyeball[0]:

            d1 = min(intersect_cornea[1], intersect_cornea[2])
            d2 = min(intersect_eyeball[1], intersect_eyeball[2])

            if d1 < d2:  # FIRST INTERSECTS CORNEA

                d = d1
                p = origin + d * v

                normal = normalize(p - eye.cornea_center)
                vnew = 1. / eye.n * (np.cross(normal, np.cross(-normal, v))) - normal * (
                    np.sqrt(1 - 1 / (eye.n ** 2) * np.dot(np.cross(normal, v), np.cross(normal, v))))

                intersect_iris = intersect_plane_line(eye.iris_center, eye.iris_normal, p, vnew,
                                                      radius=[eye.iris_radius])
                intersect_pupil = intersect_plane_line(eye.pupil_center, eye.pupil_normal, p, vnew,
                                                       radius=[eye.pupil_radius])

                if intersect_pupil[0]:
                    co = [0.0, 0.0, 0.0]
                elif intersect_iris[0]:
                    co = [0.0, 1.0, 1.0]
                else:
                    co = [0.0, 0.0, 1.0]

            else:  # FIRST INTERSECTS EYEBALL

                co = [0.4, 0.4, 0.4]

        elif intersect_cornea[0] and not intersect_eyeball[0]:  # ONLY INTERSECTS CORNEA

            d = min(intersect_cornea[1], intersect_cornea[2])
            p = origin + d * v
            normal = normalize(p - eye.cornea_center)
            vnew = 1 / eye.n * (np.cross(normal, np.cross(-normal, v))) - normal * (
                np.sqrt(1 - 1 / (eye.n ** 2) * np.dot(np.cross(normal, v), np.cross(normal, v))))

            intersect_iris = intersect_plane_line(eye.iris_center, eye.iris_normal, p, vnew,
                                                  radius=[eye.iris_radius])
            intersect_pupil = intersect_plane_line(eye.pupil_center, eye.pupil_normal, p, vnew,
                                                   radius=[eye.pupil_radius])

            if intersect_pupil[0]:
                co = [0.0, 0.0, 0.0]
            elif intersect_iris[0]:
                co = [0.0, 1.0, 1.0]
            else:
                co = [0.0, 0.0, 1.0]

        elif not intersect_cornea[0] and intersect_eyeball[0]:  # ONLY INTERSECTS EYEBALL
            co = [0.4, 0.4, 0.4]
        else:  # DOES NOT INTERSECT EYEBALL NOR CORNEA
            co = [0.9, 0.9, 0.9]
        return co

    def draw_image(self):

        self.create_texture()
        glPushMatrix()
        glTranslate(-self.X / 2., -self.Y / 2., 0)
        glScale(self.X, self.Y, 1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glEnable(GL_TEXTURE_2D)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_QUADS)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 1.0, 0.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(.0, 1.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(0, 0, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()

    def create_texture(self):

        image = PIL.Image.fromarray((255*self.image[::-1, :].astype(np.float)/self.image[::-1, :].max()).astype(np.uint8))
        ix = image.size[0]
        iy = image.size[1]
        image = image.tobytes("raw", "RGBX", 0, -1)

        glGenTextures(1, texture)
        glBindTexture(GL_TEXTURE_2D, texture)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    def fit_ellipse(self, draw=True):

        try:
            e_, contour = fit_ellipse(self.image, offsets=[0, 0])
        except:
            return False

        # THIS IS AN ADHOC CORRECTION FOR THE PIXELATION

        center_x = e_.center[0] * self.dx - self.X/2 + self.dx/2.
        center_y = e_.center[1] * self.dy - self.Y/2 + self.dy/2.
        major_radius = e_.major_radius * self.dx + self.dx/2.
        minor_radius = e_.minor_radius * self.dx + self.dy/2.
        angle = e_.angle

        if major_radius < 0.001 or minor_radius < 0.001:
            return False

        self.ellipse = Ellipse(np.array((center_x, center_y)), major_radius, minor_radius, angle)
        self.ellipse_list.append(self.ellipse)

        if draw:
            for p in contour:
                self.image[p[1], p[0]] = [1.0, 1.0, 0]

        contour = contour.astype(np.float)
        for i in range(len(contour)):
            contour[i][0] = (float(contour[i][0])/float(self.pixels_x))*self.X-self.X/2.0+self.dx/2.
            contour[i][1] = (float(contour[i][1])/float(self.pixels_y))*self.Y-self.Y/2.0+self.dy/2.
        self.ellipse_contour_list.append(contour)

        return True

    def into_world_coordinates(self, p):

        p0 = p[0] / self.pixels_x * self.X - self.X / 2.
        p1 = p[1] / self.pixels_y * self.Y - self.Y / 2.
        p2 = 0

        return np.asarray([p0, p1, p2])

    def get_unprojected_circles(self, eye, make_image=False, fit_ellipse=False, success=False):

        if make_image:
            self.make_image(eye)
        if fit_ellipse:
            success = self.fit_ellipse()

        if success:
            circle0, circle1 = unproject_ellipse(self.ellipse, 3, self.focal_length)

            if not np.isnan(circle0.center[0]):
                m0 = np.dot(eye.gaze_vector, circle0.normal)
                m1 = np.dot(eye.gaze_vector, circle1.normal)
                if m0 > m1:
                    self.circles = [circle0, circle1]
                else:
                    self.circles = [circle1, circle0]
                self.circles_list.append(self.circles)

    def intersect_projected_circle_normals(self,start=0):

        M1 = np.zeros((3,3))
        M2 = np.zeros((3,1))

        for circles in self.circles_list[start:]:

            n_tilde = np.array([project_vector_into_image_plane(circles[0].normal, np.array([0, 0, 1]))])
            p_tilde = np.array([project_point_into_image_plane(circles[0].center, self.focal_length)])

            M1 += np.eye(3)-np.dot(n_tilde.T,n_tilde)
            M2 += np.dot(np.eye(3) - np.dot(n_tilde.T, n_tilde), p_tilde.T)

        c_tilde = np.dot(np.linalg.inv(M1),M2)
        print(n_tilde.shape,p_tilde.shape,c_tilde.shape)
        return c_tilde

    def open_video_stream(self, file_, readall=True):
        self.video_capture = cv2.VideoCapture(file_)
        if readall:
            self.all_frames = []
            success=True
            while success:
                success, image = self.video_capture.read()
                if success:
                    self.all_frames.append(image.copy())
            print(len(self.all_frames))

    def read_new_frame(self, fromlist=True, randomized=True, idx=None):
        if not fromlist:
            success, self.image = self.video_capture.read()
            if not success:
                print("Could not read image from stream!")
            else:
                self.counter += 1
                self.frame = Frame(0.0, self.image, self.counter)
        else:
            try:
                if randomized:
                    idx = np.random.randint(len(self.all_frames))
                self.image = self.all_frames[idx].copy()
                self.frame = Frame(0.0, self.image,idx)
            except:
                print("Cannot read frame!")

    def detect_pupil_real_image(self):

        im_X, im_Y, ch = self.image.shape
        detection_result  = self.pupil_detector.detect(self.frame, Roi(self.image.shape), False)
        angle = detection_result["ellipse"]["angle"]*np.pi/180.0+np.pi/2
        a_minor, a_major = detection_result["ellipse"]["axes"]
        center_x, center_y = detection_result["ellipse"]["center"]

        if a_major>20:

            a_major /= (2*im_X)/self.Y
            a_minor /= (2*im_Y)/self.X
            center_x /= im_Y/self.X
            center_x -= self.X/2.
            center_y /= im_X/self.Y
            center_y -= self.Y/2

            self.ellipse = Ellipse(np.asarray([center_x,center_y]),a_minor,a_major,angle)
            self.ellipse_list.append(self.ellipse)

            roi_x, roi_y, _, __ = detection_result["roi"]
            liste = []
            for p in detection_result['edges']:
                XX = (float(p[0]+roi_x)/float(self.pixels_x))*self.X-self.X/2.0+self.dx/2.
                YY = (float(p[1]+roi_y)/float(self.pixels_y))*self.Y-self.Y/2.0+self.dy/2.
                liste.append([XX,YY])
            self.ellipse_contour_list.append(liste)

            for p in detection_result['edges']:
                    self.image[p[1]+roi_y,p[0]+roi_x] = 255

            return True

        else:

            self.pupil_detector = pupil_detectors.Detector_2D()
            return False


if __name__=="__main__":

    import time

    initial = 10
    steps = 3
    factor = 4
    print(time.time())
    eye = Eye(n=1.3375)
    camera = Camera(pixels_x=initial*factor**steps, pixels_y=initial*factor**steps)
    eye.rotate_eye(np.pi/4, axis='y')
    camera.make_image_iterative(eye, initial=initial, steps=steps, factor=factor)
    print(time.time())
    plt.imshow(camera.image)
    plt.show()

    initial = 10
    steps = 6
    factor = 2
    print(time.time())
    eye = Eye(n=1.3375)
    camera = Camera(pixels_x=initial*factor**steps, pixels_y=initial*factor**steps)
    eye.rotate_eye(np.pi / 4, axis='y')
    camera.make_image_iterative(eye, initial=initial, steps=steps, factor=factor)
    print(time.time())
    plt.imshow(camera.image)
    plt.show()
    # print(time.time())
    # camera.make_image(eye)
    # print(time.time())
    # plt.imshow(camera.image)
    # plt.show()