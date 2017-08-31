import numpy as np
from scipy import interpolate
import glob
import matplotlib.pyplot as plt


def parse_image_filename(file_,type_=1):

    if type_==0:
        pars = file_.split("/")[-1].split("_")
        phi, theta, r = float(pars[2]),float(pars[3]), float(pars[4])
        return phi, theta, r
    if type_==2:
        pars = file_.split("/")[-1].split("_")
        phi, theta, r = float(pars[1]),float(pars[2]), float(pars[3])
        return phi, theta, r

    if type_==1:
        pars = {}
        for entry in file_.split("/")[-1].split("_")[1:-1]:
            pars[entry[0]]=float(entry[1:])

        phi, theta, r = pars['p'],pars['t'],pars['r']
        x, y, z = pars['x'], pars['y'], pars['z']
        n = pars['n']
        res = pars['i']
        X, Y = pars['X'], pars['Y']

        return phi, theta, r, x, y, z, n, res, X, Y



class TrajectoryRandom(object):
    def __init__(self, mu_phi=0, mu_theta=0, mu_r=0, phi_range=(0, np.pi / 4.),
                 theta_range=(0, 2 * np.pi), r_range=(2, 3.5)):
        self.mu_phi = mu_phi
        self.mu_theta = mu_theta
        self.mu_r = mu_r
        self.phi_range = phi_range
        self.theta_range = theta_range
        self.r_range = r_range

    def next(self):
        phi = self.mu_phi + np.random.uniform(self.phi_range[0], self.phi_range[1])
        theta = self.mu_theta + np.random.uniform(self.theta_range[0], self.theta_range[1])
        r = self.mu_r + np.random.uniform(self.r_range[0], self.r_range[1])

        return phi, theta, r


class TrajectorySmooth(object):
    def __init__(self, N=100, end=20, phi_points=20, theta_points=20, r_points=20):

        self.N = N

        phi = np.random.uniform(-np.pi / 4, np.pi / 4, phi_points)
        theta = np.random.uniform(-np.pi / 4, np.pi / 4, theta_points)
        r = np.random.uniform(2, 5, r_points)

        xphi = np.linspace(0, end, phi_points)
        tckphi = interpolate.splrep(xphi, phi, s=2)

        xtheta = np.linspace(0, end, theta_points)
        tcktheta = interpolate.splrep(xtheta, theta, s=2)

        xr = np.linspace(0, end, r_points)
        tckr = interpolate.splrep(xr, r, s=2)

        self.t = np.linspace(0, end, N)
        self.phi = interpolate.splev(self.t, tckphi, der=0)
        self.theta = interpolate.splev(self.t, tcktheta, der=0)
        self.r = interpolate.splev(self.t, tckr, der=0)
        self.counter = 0

    def next(self, step=1):

        self.counter += step
        if self.counter > self.N - 1:
            self.counter = self.N - 1
        if self.counter < 0:
            self.counter = 0

        phi = np.clip(self.phi[self.counter], -np.pi / 4, np.pi / 4)
        theta = np.clip(self.theta[self.counter], -np.pi / 4, np.pi / 4)
        r = np.clip(self.r[self.counter], 2, 5)

        return phi, theta, r


class TrajectoryLevy(object):
    def __init__(self, N=1000, wt=10, D=0.003, r_points=10):

        self.N = N
        self.D = D
        self.wt = wt

        self.jump_toggle = False
        self.steady_toggle = True

        self.phi, self.theta = np.random.uniform(-np.pi / 4, np.pi / 4, 2)
        self.r = 3

        self.time = 0
        self.counter = 0
        self.jump_time = max(5, np.random.exponential(scale=self.wt))
        self.jump_duration = np.round(np.random.randint(5, 15))

        self.r_points = r_points
        self.t = np.linspace(0, N, N + 1)
        xr = np.linspace(0, N, self.r_points)
        self.r_ = np.random.uniform(2, 5, self.r_points)
        tckr = interpolate.splrep(xr, self.r_, s=1)
        self.r_spline = interpolate.splev(self.t, tckr, der=0)
        self.r_time = 0

    def next(self, steps=1):

        for _ in range(steps):
            if self.time < self.jump_time and self.steady_toggle:
                self.phi += np.random.normal(0, self.D)
                self.theta += np.random.normal(0, self.D)
                self.time += 1
                self.r_time += 1

            if self.time >= self.jump_time and not self.jump_toggle:
                self.steady_toggle = False
                self.jump_toggle = True
                self.time = 0
                self.start_phi = self.phi
                self.start_theta = self.theta
                self.end_phi, self.end_theta = np.random.uniform(- np.pi / 4, np.pi / 4, 2)

            if self.jump_toggle and self.time < self.jump_duration:
                self.phi = self.start_phi + self.time / self.jump_duration * (self.end_phi - self.start_phi)
                self.theta = self.start_theta + self.time / self.jump_duration * (self.end_theta - self.start_theta)
                self.time += 1
                self.r_time += 1

            if self.jump_toggle and self.time >= self.jump_duration:
                self.jump_toggle = False
                self.steady_toggle = True
                self.time = 0
                self.jump_time = max(5, np.random.exponential(scale=self.wt))
                self.jump_duration = np.round(np.random.randint(5, 15))

            if self.r_time == self.N:
                xr = np.linspace(0, self.N, self.r_points)
                temp = self.r_[-1]
                self.r_ = np.random.uniform(2, 5, self.r_points)
                self.r_[0] = temp
                tckr = interpolate.splrep(xr, self.r_, s=1)
                self.r_spline = interpolate.splev(self.t, tckr, der=0)
                self.r_time = 0

        return self.phi, self.theta, np.clip(self.r_spline[self.r_time], 3, 4.5), ""


class TrajectoryPreRendered(object):
    def __init__(self, dir_=None, randomized_=False, file_list=False, type_=1):
        if not file_list:
            self.dir_ = dir_
            self.file_list = glob.glob(dir_ + "*.npz")
        else:
            self.file_list = list(reversed(file_list))
        self.randomized_ = randomized_
        self.type_ = type_

    def next(self):

        if not self.randomized_:
            file_ = self.file_list.pop()
        else:
            file_ = np.random.choice(self.file_list)

        if self.type_==1:
            phi, theta, r, x, y, z, n, res, X, Y = parse_image_filename(file_, type_=self.type_)
        if self.type_==0:
            phi, theta, r = parse_image_filename(file_, type_=self.type_)
        if self.type_ == 2:
            phi, theta, r = parse_image_filename(file_, type_=self.type_)


        with np.load(file_) as data:
            img = data['arr_0']

        return phi, theta, r, img, file_


if __name__ == "__main__":
    traj = Trajectory()
    print(traj.next())
