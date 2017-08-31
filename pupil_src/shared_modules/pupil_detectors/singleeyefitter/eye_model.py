import numpy as np
from .geometry import *
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from .projections import *
from joblib import Parallel, delayed
from random import sample

def parse_history(out):
    lines = out.split("new_iteration")
    steps = len(lines)
    n_pars = len(lines[0].split())
    print(steps,n_pars)
    history = np.zeros((steps, n_pars))
    for i in range(steps):
        temp = lines[i].split()
        for j in range(n_pars):
            history[i][j] = float(temp[j])
    return history

class EyeModel(object):

    def __init__(self, ellipse_list=False, edge_list=False, n=1.0, init=True, focal_length=10, re=12.0, rc=7.3, ri=5.9):

        self.ellipse_list = None
        self.unprojected_ellipses = []
        self.edge_list = None
        self.c_tilde = np.array([0.0, 0.0, 0.0])
        self.c = np.array([0.0, 0.0, 50.0])
        self.c_fit = np.array([0.0, 0.0, 50.0])
        self.fit_parameters = None
        self.history = None
        self.radii_fit = None
        self.eye_param = [re, rc, ri]
        self.dp = np.sqrt(self.eye_param[0]**2 - self.eye_param[2]**2) # dp
        self.sphere_radius = None
        self.consistent_pupil_list = []
        self.pruned_edges = []
        self.pupil_radius_set = 0.5
        self.n = n
        self.focal_length = focal_length
        self.correct_p = None
        self.gv_fit = None

        if init:
            if not (ellipse_list and edge_list):
                print("No data for initialization!")
            else:
                self.ellipse_list = list(ellipse_list)
                self.unprojected_ellipses = []
                self.edge_list = list(edge_list)
                self.unproject_ellipses()
                self.find_projected_sphere_center()
                self.disambiguate_all_unprojected_ellipses()
                self.estimate_sphere_radius()
                self.scale_sphere(self.dp)
                self.consistent_pupil_estimates()

    def prune_contours(self, N=26):
        self.pruned_edges = []
        for c in self.edge_list:
            self.pruned_edges.append(sample(c,26))

    def unproject_ellipses(self):
        self.unprojected_ellipses = []
        toggle = False
        for idx, e in enumerate(self.ellipse_list):
            result = unproject_ellipse(e, self.pupil_radius_set, self.focal_length)
            if result:
                self.unprojected_ellipses.append(result)
            else:
                to_delete = idx
                toggle = True
        if toggle:
            print("Deleting")
            del self.ellipse_list[to_delete]

    def find_projected_sphere_center(self):
        M1 = np.zeros((3, 3))
        M2 = np.zeros((3, 1))

        for circles in self.unprojected_ellipses:
            n_tilde = np.array([project_vector_into_image_plane(circles[0].normal, np.array([0.0, 0.0, 1.0]))])
            p_tilde = np.array([project_point_into_image_plane(circles[0].center, self.focal_length)])

            M1 += np.eye(3) - np.dot(n_tilde.T, n_tilde)
            M2 += np.dot(np.eye(3) - np.dot(n_tilde.T, n_tilde), p_tilde.T)

        c_tilde = np.dot(np.linalg.inv(M1), M2)

        self.c_tilde = c_tilde
        self.c = unproject_point(self.c_tilde, z=42.0, focal_length=self.focal_length)
        # print(self.c)

    def disambiguate_all_unprojected_ellipses(self):
        temp = []
        for circles in self.unprojected_ellipses:
            n_tilde = np.array([project_vector_into_image_plane(circles[0].normal, np.array([0, 0, 1]))])
            p_tilde = np.array([project_point_into_image_plane(circles[0].center, self.focal_length)])
            if np.dot(self.c_tilde[0] - p_tilde[0], n_tilde[0]) < 0:  # THIS IS OPPOSITE TO THE PAPER,
                # BUT I THINK IT IS A TYPO
                temp.append([circles[0], circles[1]])
            else:
                temp.append([circles[1], circles[0]])
        self.unprojected_ellipses = list(temp)

    def disambiguate_unprojected_ellipse(self, unprojected_ellipse):
        circle0, circle1 = unprojected_ellipse
        n_tilde = np.array([project_vector_into_image_plane(circle0.normal, np.array([0.0, 0.0, 1.0]))])
        p_tilde = np.array([project_point_into_image_plane(circle0.center, self.focal_length)])
        if np.dot(self.c_tilde[0] - p_tilde[0], n_tilde[0]) < 0:  # THIS IS OPPOSITE TO THE PAPER,
            # BUT I THINK IT IS A TYPO
            return circle0
        else:
            return circle1

    def estimate_sphere_radius(self):

        average_radius = 0
        counter = 0

        for entry in self.unprojected_ellipses:
            pupil = entry[0]

            M1 = np.zeros((3, 3))
            M2 = np.zeros((3, 1))

            o1 = np.expand_dims(self.c, axis=0)
            l1 = normalize(np.expand_dims(pupil.normal, axis=0))
            M1 += np.eye(3) - np.dot(l1.T, l1)
            M2 += np.dot(np.eye(3) - np.dot(l1.T, l1), o1.T)

            o2 = np.expand_dims([0, 0, -self.focal_length], axis=0)
            l2 = normalize(np.expand_dims(pupil.center, axis=0) - o2)
            M1 += np.eye(3) - np.dot(l2.T, l2)
            M2 += np.dot(np.eye(3) - np.dot(l2.T, l2), o2.T)

            intersection = np.reshape(np.dot(np.linalg.inv(M1), M2), (3))

            average_radius += np.linalg.norm(intersection - self.c)
            counter += 1.0

        self.sphere_radius = average_radius / counter

    def consistent_pupil_estimates(self):
        self.consistent_pupil_list = []
        for entry in self.unprojected_ellipses:

            pupil = entry[0]
            o = np.asarray([0, 0, -self.focal_length])
            l = normalize(pupil.center - o)

            success = intersect_sphere_line(self.c, self.sphere_radius, o, l)

            if success[0]:
                d1 = success[1]
                d2 = success[2]
                p1 = o + d1 * l
                p2 = o + d2 * l
                if np.linalg.norm(p1 - pupil.center) < np.linalg.norm(p2 - pupil.center):
                    pprime = p1
                else:
                    pprime = p2
                normal = normalize(pprime - self.c)

            else:
                temp = np.array(o + np.dot(self.c - o, l) * l)  # CLOSEST POINT ON RAY TO SPHERE
                normal = normalize(temp-self.c)
                pprime = self.c+normal*self.sphere_radius

            radius = (pprime[2] + self.focal_length) / (pupil.center[2] + self.focal_length) * self.pupil_radius_set
            self.consistent_pupil_list.append(Circle(pprime, normal, radius))

    def scale_sphere(self, radius):

        scale = radius / self.sphere_radius

        self.sphere_radius *= scale
        origin = np.array([0.0, 0.0, -self.focal_length])
        self.c = scale*(self.c-origin)+origin

        print(self.c)

        for pupil in self.consistent_pupil_list:
            pupil.center = scale*(pupil.center-origin)+origin
            pupil.radius *= scale

    def minimize_edge_distance_ray_tracing(self):

        n = len(self.consistent_pupil_list)
        N = 3 + n * 3
        p0 = np.zeros(N)
        p0[0] = self.c[0]
        p0[1] = self.c[1]
        p0[2] = self.c[2]

        for i, pupil in enumerate(self.consistent_pupil_list):
            phi, theta = spherical_coordinates(pupil.center, self.c)
            p0[3 + i * 3] = phi
            p0[3 + i * 3 + 1] = theta
            p0[3 + i * 3 + 2] = pupil.radius

        N = int((len(p0) - 3) / 3)

        def energy_functional_parallel(p, plot=False):

            circles = []
            for i_ in range(N):
                radius = p[3 + i_ * 3 + 2]
                normal = from_spherical(p[3 + i_ * 3 + 0], p[3 + i_ * 3 + 1])
                center = p[0:3] + self.sphere_radius * normal
                circle = Circle(center, normal, radius)
                circles.append(circle)

            if self.n == 1.0:
                ellipses = Parallel(n_jobs=N)(delayed(project_circle_into_image_plane)(circle, focal_length=self.focal_length)
                                              for circle in circles)
            else:
                ellipses = Parallel(n_jobs=N)(delayed(project_circle_into_image_plane_refraction)(circle,
                                                                                                  p[0:3],
                                                                                                  n=self.n,
                                                                                                  focal_length=self.focal_length,
                                                                                                  initial=10,
                                                                                                  steps=5,
                                                                                                  factor=2)
                                              for circle in circles)
            cost = []

            ax = None
            if plot:
                fig = plt.figure()
                ax = plt.gca()

            for i in range(N):
                distances = approximate_distance_points_to_ellipse(self.edge_list[i], ellipses[i], plot=plot, ax = ax)
                cost += list(distances)

            plt.show()

            # CURRENT COST AND CENTER
            print("Current cost:", np.sum(np.asarray(cost) ** 2), p[:3])

            return np.sum(np.asarray(cost) ** 2)/len(cost)

        energy_functional_parallel(self.correct_p, plot=True)
        p1 = minimize(energy_functional_parallel, p0, method='Nelder-Mead', options={'maxfev': 2000})
        energy_functional_parallel(p1.x, plot=True)
        self.c_fit = p1.x[0:3]
        self.radii_fit = p1.x[5::3]
        self.fit_parameters = p1.x

    def minimize_edge_distance_ray_tracing_fixed_sphere(self, center_= [0, 0, 50]):

        self.c = center_
        self.consistent_pupil_estimates()

        n = len(self.consistent_pupil_list)
        p0 = np.zeros(n * 3)

        for i, pupil in enumerate(self.consistent_pupil_list):
            phi, theta = spherical_coordinates(pupil.center, self.c)
            p0[i * 3] = phi
            p0[i * 3 + 1] = theta
            p0[i * 3 + 2] = pupil.radius

        NN = int(len(p0) / 3)

        def energy_functional_parallel(p, plot=False):

            circles = []
            for i_ in range(NN):
                radius = p[i_ * 3 + 2]
                normal = from_spherical(p[i_ * 3 + 0], p[i_ * 3 + 1])
                center = center_ + self.sphere_radius * normal
                circle = Circle(center, normal, radius)
                circles.append(circle)

            if self.n == 1.0:
                ellipses = Parallel(n_jobs=1)(delayed(project_circle_into_image_plane)(circle, focal_length=self.focal_length)
                                              for circle in circles)
            else:
                ellipses = Parallel(n_jobs=1)(delayed(project_circle_into_image_plane_refraction)(circle,
                                                                                                  center_,
                                                                                                  n=self.n,
                                                                                                  focal_length=self.focal_length,
                                                                                                  initial=10,
                                                                                                  steps=5,
                                                                                                  factor=2)
                                              for circle in circles)
            cost = []

            ax = None
            if plot:
                fig = plt.figure()
                ax = plt.gca()

            for i in range(NN):
                distances = approximate_distance_points_to_ellipse(self.edge_list[i], ellipses[i], plot=plot, ax = ax)
                cost += list(distances)

            plt.show()

            # CURRENT COST AND CENTER
            print("Current cost:", np.sum(np.asarray(cost) ** 2), p[:3])

            return np.sum(np.asarray(cost) ** 2)/len(cost)

        p1 = minimize(energy_functional_parallel, p0, method='Nelder-Mead', options={'maxfev': 2000})
        energy_functional_parallel(p1.x, plot=True)
        self.fit_parameters = p1.x

        return p1

    def minimize_edge_distance_tangent_plane_fixed_sphere(self, center_= [0, 0, 50], nn=-1):
        self.c = np.asarray(center_)
        self.consistent_pupil_estimates()

        if nn < 0:
            nn = len(self.consistent_pupil_list)
        N = nn * 3
        p0 = np.zeros(N)

        for i, pupil in enumerate(self.consistent_pupil_list):

            # phi, theta = spherical_coordinates(pupil.center, self.c)

            p0[i * 3] = self.correct_p[3+3*i]
            p0[i * 3 + 1] = self.correct_p[3+3*i+1]
            p0[i * 3 + 2] = self.correct_p[3+3*i+2]

        NN = int(len(p0) / 3)

        def energy_functional_parallel(p, plot=False):

            scale_factor = 0.01 * np.linalg.norm(np.array([0, 0, -self.focal_length]) - np.asarray(center_))

            cost = []

            for i_ in range(NN):

                radius = p[i_ * 3 + 2]
                normal = from_spherical(p[i_ * 3 + 0], p[i_ * 3 + 1])
                center = np.array(center_) + self.sphere_radius * normal
                pupil = Circle(center, normal, radius)
                upprojected_points = upproject_image_pointlist_into_tangent_plane(self.edge_list[i_], pupil, center_, n=self.n, focal_length=self.focal_length, plot=plot)
                distances = [((np.linalg.norm(p__-pupil.center)-pupil.radius)/scale_factor)**2 for p__ in upprojected_points]
                cost += distances

            # CURRENT COST AND CENTER
            final_cost = np.sum(np.asarray(cost))  # /len(cost)
            print("Current cost:", final_cost)
            return final_cost

        print("Initial guess:", p0)
        p1 = minimize(energy_functional_parallel, p0, method='L-BFGS-B',
                      options={'maxiter': 10000, 'ftol': 10E-7, 'gtol':10E-7})

        # SET INTERNAL VARIABLES TO FIT VALUES
        self.c_fit = center_
        self.radii_fit = p1.x[2::3]
        self.fit_parameters = p1.x

        return p1

    def minimize_edge_distance_tangent_plane_fixed_sphere_cpp(self, center_= [0, 0, 50], nn=-1):
        import subprocess

        self.c = np.asarray(center_)
        self.consistent_pupil_estimates()

        if nn < 0:
            nn = len(self.consistent_pupil_list)

        cmd = ["%i" %nn,
               "%.12f" %self.n,
               "%.12f" %self.c[0],
               "%.12f" %self.c[1],
               "%.12f" %self.c[2]]

        for i in range(nn):
            cmd += ["%.12f" %self.correct_p[3+3*i]]
            cmd += ["%.12f" %self.correct_p[3+3*i+1]]
            cmd += ["%.12f" %self.correct_p[3+3*i+2]]
            for p in self.edge_list[i][:26]:
                cmd.append("%.12f" % p[0])
                cmd.append("%.12f" % p[1])

        try:
            temp = subprocess.check_output(["/cluster/Kai/refraction/cpp/potential"] + cmd)
            print(" " .join(["/cluster/Kai/refraction/cpp/potential"] + cmd))
        except:
            temp = "100 100 100 100 100"
        res = temp.split()
        p1 = [float(s) for s in res]

        # SET INTERNAL VARIABLES TO FIT VALUES
        self.c_fit = center_
        self.radii_fit = p1[2::3]
        self.fit_parameters = p1

        return p1

    def minimize_edge_distance_tangent_plane_all_parameters(self):
        n = len(self.consistent_pupil_list)
        N = 3 + n * 3
        p0 = np.zeros(N)
        p0[0] = self.c[0]
        p0[1] = self.c[1]
        p0[2] = self.c[2]

        for i, pupil in enumerate(self.consistent_pupil_list):
            phi, theta = spherical_coordinates(pupil.center, self.c)
            p0[3 + i * 3] = phi
            p0[3 + i * 3 + 1] = theta
            p0[3 + i * 3 + 2] = pupil.radius

        N = int((len(p0) - 3) / 3)

        history = []

        def callback(p_):
            history.append(list(p_))
            np.save("/cluster/Kai/refraction/results/history.npy", np.asarray(history))

        def energy_functional_parallel(p, plot=False):

            scale_factor = 0.01 * np.linalg.norm(np.array([0, 0, -self.focal_length]) - np.asarray(p[:3]))

            cost = []
            for i_ in range(N):
                radius = p[3 + i_ * 3 + 2]
                normal = from_spherical(p[3 + i_ * 3 + 0], p[3 + i_ * 3 + 1])
                center = p[:3] + self.sphere_radius * normal
                pupil = Circle(center, normal, radius)

                upprojected_points = upproject_image_pointlist_into_tangent_plane(self.edge_list[i_], pupil, p[:3], n=self.n, focal_length=self.focal_length, plot=plot)

                distances = [((np.linalg.norm(p__ - pupil.center) - pupil.radius) / scale_factor) ** 2 for p__ in upprojected_points]
                cost += distances

            # CURRENT COST AND CENTER
            final_cost = np.sum(np.asarray(cost)) #/len(cost)

            print("Cost:", final_cost)
            print("Center:", p[:3])

            return final_cost

        #p1 = minimize(energy_functional_parallel, p0, method='L-BFGS-B', callback=callback, options={'maxiter':5000, 'ftol': 0.00000001})
        p1 = minimize(energy_functional_parallel, p0, method='L-BFGS-B', options={'maxfun': 1000000, 'maxiter': 2000, 'ftol': 10E-8, 'gtol': 10E-8})

        # SET INTERNAL VARIABLES TO FIT VALUES
        self.c_fit = p1.x[0:3]
        self.radii_fit = p1.x[5::3]
        self.fit_parameters = p1.x

        print(p1)

        return p1, history

    def minimize_edge_distance_tangent_plane_all_parameters_cpp(self):

        focal_length = self.focal_length
        import subprocess

        n = len(self.consistent_pupil_list)

        pars = []
        pars.append("%i" % n)
        pars.append("%.12f" % self.n)
        for i, pupil in enumerate(self.consistent_pupil_list):
            pars.append("%i" % min(30,len(self.edge_list[i])))
        pars.append("%.12f" % self.c[0])
        pars.append("%.12f" % self.c[1])
        pars.append("%.12f" % self.c[2])
        pars.append("%.12f" % self.eye_param[0])
        pars.append("%.12f" % self.eye_param[1])
        pars.append("%.12f" % self.eye_param[2])

        for i, pupil in enumerate(self.consistent_pupil_list):
            phi, theta = spherical_coordinates(pupil.center, self.c)
            pars.append("%.12f" % phi)
            pars.append("%.12f" % theta)
            pars.append("%.12f" % pupil.radius)
            np.random.shuffle(self.edge_list[i])
            for p in self.edge_list[i][:min(30, len(self.edge_list[i]))]:
                pars.append("%.12f" % p[0])
                pars.append("%.12f" % p[1])

        print(" ".join(["/cluster/Kai/refraction/cpp/sphere_optimization_with_eye_params"] + pars))
        temp = subprocess.check_output(["/cluster/Kai/refraction/cpp/sphere_optimization_with_eye_params"] + pars).decode("utf-8")
        self.history = parse_history(temp)
        p1 = self.history[-1]

        # SET INTERNAL VARIABLES TO FIT VALUES
        self.c_fit = p1[0:3]
        self.eye_param[1] = self.history[-1][3]
        self.eye_param[2] = self.history[-1][4]
        self.dp = np.sqrt(self.eye_param[0]**2 - self.eye_param[2]**2) # dp

        print(self.c_fit)
        print(self.eye_param)
        self.radii_fit = p1[7::3]
        self.fit_parameters = p1

        return p1

    def gaze_vector(self, ellipse, mode='fit', c_given=[0, 0, 50]):

        if mode=='fit':
            c_ = self.c_fit
        elif mode=='unoptimized':
            c_ = self.c
        elif mode=='perfect':
            c_ = np.array([0, 0, 50], dtype=np.float)
        elif mode=="manual":
            c_ = np.array(c_given)

        unprojected = unproject_ellipse(ellipse, self.pupil_radius_set, self.focal_length)
        circle = self.disambiguate_unprojected_ellipse(unprojected)
        o = np.asarray([0.0, 0.0, -self.focal_length])
        l = normalize(circle.center - np.asarray([0.0, 0.0, -self.focal_length]))

        success = intersect_sphere_line(c_, self.sphere_radius, o, l)
        if success[0]:
            d = min(success[1], success[2])
            p = o + d * l  #INTERSECTION POINT OF RAY AND SPHERE
            gaze_vector = normalize(p - c_)
        else:
            NP = np.array(o+np.dot(c_-o,l)*l) #CLOSEST POINT ON RAY TO SPHERE
            gaze_vector = normalize(NP - c_)

        return gaze_vector

    def gaze_vector_refraction_ray_tracing(self, ellipse, edges, initial=20, factor=2, steps=4):

        def energy_functional(p, plot=False):

            print("Current guess:", p)

            upprojected_points = upproject_image_pointlist_into_tangent_plane(self.edge_list[i_], pupil, p[0:3], n=self.n,  focal_length = self.focal_length, plot=plot)

            # SETUP EYE
            eye = Eye(n=self.n)
            eye.rotate_to_gaze_vector(p[:3])
            eye.move_to_point(self.c_fit)
            eye.pupil_radius = p[3]

            # SETUP CAMERA
            resolution = initial * factor ** steps
            camera = Camera(pixels_x=resolution, pixels_y=resolution)

            try:
                camera.make_image_iterative(eye, initial=initial, steps=steps, factor=factor)
                camera.fit_ellipse()

                if plot:
                    plt.imshow(camera.image[::-1, :], extent=[-2.5, 2.5, -2.5, 2.5])
                    x, y = ellipse_points(camera.ellipse_list[-1])
                    plt.plot(x, y)
                    plt.plot(edges[:, 0], edges[:, 1], "o")
                    plt.show()

                distances = approximate_distance_points_to_ellipse(edges, camera.ellipse_list[-1], plot=False)
                cost = np.sum(np.array(list(distances) + [
                    0.1 * (1.0 - np.linalg.norm(p[:3]))]) ** 2)  # WE ADD A TERM TO ENFORCE NORMALIZATION

                print(cost)

                return np.asarray(cost)/np.sqrt(len(cost))

            except:
                distances = 100 * np.ones(len(edges))
                return (len(distances) + 1.0) * 1000

        p0 = list(self.gaze_vector(ellipse, mode='perfect')) + [2.5]  # INITIAL GUESS FOR GV AND RADIUS
        p1 = minimize(energy_functional, p0, method='Nelder-Mead', options={'maxiter': 50})
        # energy_functional(p1.x, plot=True)

        return normalize(p1.x[:3]), p1.x[3]

    def gaze_vector_refraction_tangent_plane(self, ellipse, edges, mode='fit'):


        def energy_functional(p, plot=False):

            #print("Current guess:", p)

            cost = []
            radius = p[2]
            normal = from_spherical(p[0], p[1])

            if mode == 'fit':
                c_ = self.c_fit
            elif mode == 'unoptimized':
                c_ = self.c
            elif mode == 'perfect':
                c_ = np.array([0 ,0, 50], dtype=np.float)

            center = c_ + self.sphere_radius * normal
            pupil = Circle(center, normal, radius)
            upprojected_points = upproject_image_pointlist_into_tangent_plane(edges, pupil, c_,
                                                                                  n=self.n,  focal_length = self.focal_length,
                                                                                  plot=plot)

            distances = [(np.linalg.norm(p__ - pupil.center) - pupil.radius) ** 2 for p__ in upprojected_points]
            cost += distances

            # CURRENT COST AND CENTER
            final_cost = np.sum(np.asarray(cost))/len(cost)
            #print(final_cost, p[:3])
            return final_cost

        p0 = list(spherical_coordinates(self.gaze_vector(ellipse, mode='fit'), self.c))+[3]  # INITIAL GUESS FOR GV AND RADIUS
        p1 = minimize(energy_functional, p0, method='L-BFGS-B')
        gv = from_spherical(p1.x[0], p1.x[1])

        return gv, p1.x[2]

    def gaze_vector_refraction_tangent_plane_cpp(self, ellipse, edges, mode='fit'):

            import subprocess

            pars = []
            pars.append("%f" % self.n)
            pars.append("%.12f" % self.c_fit[0])
            pars.append("%.12f" % self.c_fit[1])
            pars.append("%.12f" % self.c_fit[2])
            pars.append("%.12f" % self.eye_param[0])
            pars.append("%.12f" % self.eye_param[1])
            pars.append("%.12f" % self.eye_param[2])

            p0 = list(spherical_coordinates(self.gaze_vector(ellipse, mode=mode), np.asarray([0.0,0.0,0.0]))) + [2.0]  # INITIAL GUESS FOR GV AND RADIUS
            #p0 = list(spherical_coordinates(np.asarray([0, 0, -1]), self.c_fit)) + [3.0]  # INITIAL GUESS FOR GV AND RADIUS
            pars += ["%.12f"%e_ for e_ in p0]

            length = len(edges)
            if length > 25:
                step = int(length / 25)
                idx = range(0, length, step)
                while len(idx) < 26:
                    randomi = np.random.randint(0, length)
                    idx = list(idx)
                    idx.append(randomi)

            for i in idx:
                pars.append("%.12f" % edges[i][0])
                pars.append("%.12f" % edges[i][1])

            cmd = ["/cluster/Kai/refraction/cpp/gaze_vector"] + pars
            print(" ".join(cmd))
            temp = subprocess.check_output(cmd)
            p1 = [float(s) for s in temp.split()[:3]]
            gv = from_spherical(p1[0], p1[1])
            #print(temp.split(b"ttt")[1])

            self.gv_fit = list(p1)

            return gv, p1[2]

