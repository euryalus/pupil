#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <algorithm> // for std::copy
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>
#include "projection.h"
#include "geometry/Sphere.h"
#include "geometry/Ellipse.h"
#include "EllipseDistanceApproxCalculator.h"
#include "DistancePointEllipse.h"
#include "utils.h"


int main()
{
    typedef ceres::Jet<double,6> T;

    singleeyefitter::Ellipse2D<T> ellipse_jet(T(28.8695),T(11.5014),T(35.0471),T(2.3969),T(0.83788*M_PI));
    std::cout << singleeyefitter::DistancePointEllipse<T>(ellipse_jet, T(-207.0), T(129.0)) << std::endl;

    singleeyefitter::Ellipse2D<double> ellipse_double(28.8695,11.5014,35.0471,2.3969,0.83788*M_PI);
    std::cout << singleeyefitter::DistancePointEllipse<double>(ellipse_double, -207.0, 129.0) << std::endl;

}
