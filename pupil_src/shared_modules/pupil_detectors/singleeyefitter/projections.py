import numpy as np
from eye import Eye
from camera import Camera
from geometry import Conic, Ellipse, intersect_sphere_line, normalize, intersect_plane_line
import matplotlib.pyplot as plt


def project_circle_into_image_plane(circle, focal_length):

    c = circle.center - np.array([0, 0, -focal_length])
    n = circle.normal
    r = circle.radius
    f = focal_length

    cn = np.dot(c, n)
    c2r2 = np.dot(c, c) - r ** 2
    ABC = (cn ** 2 - 2.0 * cn * (c * n) + c2r2 * (n ** 2))
    F = 2.0 * (c2r2 * n[1] * n[2] - cn * (n[1] * c[2] + n[2] * c[1]))
    G = 2.0 * (c2r2 * n[2] * n[0] - cn * (n[2] * c[0] + n[0] * c[2]))
    H = 2.0 * (c2r2 * n[0] * n[1] - cn * (n[0] * c[1] + n[1] * c[0]))
    conic = Conic(ABC[0], H, ABC[1], G * f, F * f, ABC[2] * f ** 2)

    disc_ = conic.discriminant()

    if disc_ < 0:

        A, B, C, D, E, F = conic.A, conic.B, conic.C, conic.D, conic.E, conic.F
        center_x = (2 * C * D - B * E) / disc_
        center_y = (2 * A * E - B * D) / disc_
        temp_ = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + disc_ * F)
        minor_axis = -np.sqrt(temp_ * (A + C - np.sqrt((A - C) ** 2 + B ** 2))) / disc_
        major_axis = -np.sqrt(temp_ * (A + C + np.sqrt((A - C) ** 2 + B ** 2))) / disc_

        if B == 0 and A < C:
            angle = 0
        elif B == 0 and A >= C:
            angle = np.pi / 2.
        else:
            angle = np.arctan((C - A - np.sqrt((A - C) ** 2 + B ** 2)) / B)

        return Ellipse(np.asarray([center_x, center_y]), minor_axis, major_axis, angle)

    else:

        return False


def project_circle_into_image_plane_refraction(circle, center, n=1.3375, focal_length=10, initial=10, steps=5, factor=2):

    # INITIAL EYE
    eye = Eye(pupil_radius=circle.radius, n=n)

    # ROTATE EYE TO GIVEN GAZE DIRECTION
    eye.rotate_to_gaze_vector(circle.normal)

    # TRANSLATE EYE TO GIVEN POSITION
    eye.move_to_point(center)

    # MAKE IMAGE AND FIT ELLIPSE
    resolution = initial*factor**steps
    camera = Camera(X=5.0, Y=5.0, pixels_x=resolution, pixels_y=resolution, focal_length=focal_length)
    camera.make_image_iterative(eye, initial=initial, steps=steps, factor=factor)
    camera.fit_ellipse()

    return camera.ellipse_list[-1]


def upproject_image_point_into_tangent_plane(p, pupil, eyeball_center, n=1.3375, focal_length=10):

        eye = Eye(n=n)
        eye.move_to_point(eyeball_center)
        eye.rotate_to_gaze_vector(pupil.normal)

        x, y = p[0], p[1]
        origin = np.asarray([0, 0, -focal_length], dtype=np.float)
        z = focal_length
        v = np.asarray([x, y, z], dtype=np.float)
        v = normalize(v)

        intersect_cornea = intersect_sphere_line(eye.cornea_center, eye.cornea_radius, origin, v)

        if intersect_cornea[0]:

            d = min(intersect_cornea[1], intersect_cornea[2])
            pt = origin + d * v

            normal = normalize(pt - eye.cornea_center)
            vnew = 1. / eye.n * (np.cross(normal, np.cross(-normal, v))) - normal * (np.sqrt(1. - 1. / (eye.n ** 2) * np.dot(np.cross(normal, v), np.cross(normal, v))))

            intersect_tangent_plane = intersect_plane_line(pupil.center, pupil.normal, pt, vnew)

            if intersect_tangent_plane[0]:
                return  np.asarray(intersect_tangent_plane[1:], dtype=np.float)
            else:
                print("Upprojection not in iris and not in tangent plane!")
                return np.array([0, 0, -1000], dtype=np.float)
        else:
            print("Point not in iris!")
            intersect_tangent_plane = intersect_plane_line(pupil.center, pupil.normal, origin, v)
            if intersect_tangent_plane[0]:
                print("Returning point in tangent plane without refraction!")
                return np.asarray(intersect_tangent_plane[1:], dtype=np.float)
            else:
                return np.array([0, 0, -1000], dtype=np.float)


def upproject_image_pointlist_into_tangent_plane(ps, pupil, eyeball_center, n=1.3375, focal_length=10, plot=False):
    result = []
    for p_ in ps:
        result.append(upproject_image_point_into_tangent_plane(p_, pupil, eyeball_center, n=n, focal_length=focal_length))
    if plot:
        for p_ in result:
            x = np.dot(p_ - pupil.center, pupil.tangent_1)
            y = np.dot(p_ - pupil.center, pupil.tangent_2)
            plt.plot(x,y,"o")
        phis = np.linspace(0, 2*np.pi, 100)
        xs = [pupil.radius * np.cos(phi) for phi in phis]
        ys = [pupil.radius * np.sin(phi) for phi in phis]
        plt.plot(xs,ys)
        plt.gca().set_aspect('equal')
        plt.show()

    return result


def upproject_image_point_into_tangent_plane_three_steps(p, pupil, eyeball_center, n=1.3375, focal_length=10):

        eye = Eye(n=n)
        eye.move_to_point(eyeball_center)
        eye.rotate_to_gaze_vector(pupil.normal)

        x, y = p[0], p[1]
        origin = np.asarray([0, 0, -focal_length], dtype=np.float)
        z = focal_length
        v = np.asarray([x, y, z], dtype=np.float)
        v = normalize(v)

        intersect_cornea = intersect_sphere_line(eye.cornea_center, eye.cornea_radius, origin, v)
        intersect_eyeball = intersect_sphere_line(eye.eyeball_center, eye.eyeball_radius, origin, v)

        # DETERMINE WHICH UPPROJECTION WILL BE DONE
        case_ = 0

        if ((not intersect_cornea[0]) and intersect_eyeball[0]):

            d = min(intersect_eyeball[1], intersect_eyeball[2])
            case_=2

        if (intersect_cornea[0] and intersect_eyeball[0]):

            d_cornea = min(intersect_cornea[1], intersect_cornea[2])
            d_eyeball = min(intersect_eyeball[1], intersect_eyeball[2])

            if d_cornea<d_eyeball:
                d = d_cornea
                case_=1
            else:
                d = d_eyeball
                case_=2

        if (intersect_cornea[0] and (not intersect_eyeball[0])):

            d = min(intersect_cornea[1], intersect_cornea[2])
            case_=1

        # DO UPPROJECTION
        if case_==2:

            pt = origin + d * v
            dist_ = np.linalg.norm(eye.pupil_center-pt)
            intersect_tangent_plane = intersect_plane_line(pupil.center, pupil.normal, pt, eye.pupil_normal)
            if intersect_tangent_plane[0]:
                p_intersect = np.asarray(intersect_tangent_plane[1:], dtype=np.float)
                v_intersect = normalize(p_intersect-eye.pupil_center)
                return eye.pupil_center+dist_*v_intersect

            else:
                print("EYEBALL UPPROJECTION DID NOT WORK!")
                return np.array([0, 0, -1000], dtype=np.float)

        if case_==1:

            pt = origin + d * v
            normal = normalize(pt - eye.cornea_center)
            vnew = 1. / eye.n * (np.cross(normal, np.cross(-normal, v))) - normal * (
                np.sqrt(1. - 1. / (eye.n ** 2) * np.dot(np.cross(normal, v), np.cross(normal, v))))
            intersect_tangent_plane = intersect_plane_line(pupil.center, pupil.normal, pt, vnew)

            if intersect_tangent_plane[0]:
                return np.asarray(intersect_tangent_plane[1:], dtype=np.float)

            else:
                print("Upprojection not in iris and not in tangent plane!")
                return np.array([0, 0, -1000], dtype=np.float)

        else:

            intersect_tangent_plane = intersect_plane_line(pupil.center, pupil.normal, origin, v)
            if intersect_tangent_plane[0]:
                print("Returning point in tangent plane without refraction!")
                return np.asarray(intersect_tangent_plane[1:], dtype=np.float)
            else:
                print("Upprojection not in iris and not in tangent plane!")
                return np.array([0, 0, -1000], dtype=np.float)


def upproject_image_pointlist_into_tangent_plane_three_steps(ps, pupil, eyeball_center, n=1.3375, focal_length=10, plot=False):
    result = []
    for p_ in ps:
        result.append(
            upproject_image_point_into_tangent_plane_three_steps(p_, pupil, eyeball_center, n=n, focal_length=focal_length))
    if plot:
        for p_ in result:
            x = np.dot(p_ - pupil.center, pupil.tangent_1)
            y = np.dot(p_ - pupil.center, pupil.tangent_2)
            plt.plot(x, y, "o")
        phis = np.linspace(0, 2 * np.pi, 100)
        xs = [pupil.radius * np.cos(phi) for phi in phis]
        ys = [pupil.radius * np.sin(phi) for phi in phis]
        plt.plot(xs, ys)
        plt.gca().set_aspect('equal')
        plt.show()

    return result



