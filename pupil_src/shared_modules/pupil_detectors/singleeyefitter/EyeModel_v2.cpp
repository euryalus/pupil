#include "EyeModel_v2.h"

#include <algorithm>
#include <future>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

#include "GazeAngleLossFunction.h"

#include "utils.h"
#include "math/intersect.h"
#include "projection.h"
#include "fun.h"

#include "mathHelper.h"
#include "math/distance.h"

namespace singleeyefitter {


EyeModel_v2::EyeModel_v2(double timestamp,  double focalLength, Vector3 cameraCenter):
    mBirthTimestamp(timestamp),
    mFocalLength(std::move(focalLength)),
    mCameraCenter(std::move(cameraCenter))
    {};

EyeModel_v2::~EyeModel_v2()
{}

int EyeModel_v2::addObservation(std::shared_ptr<Detector2DResult>& observation2D, bool prepare_toggle)
{

    // If there has been a reset, set a new birth timestamp of the model
    if (mBirthTimestamp == -1){

        mBirthTimestamp = observation2D->timestamp;

    }

    // if set to true, recalculate make conversion of coordinates
    if (prepare_toggle){

        prepareObservation(observation2D);

    };

    // Adding pointer to new observation
    auto newObservationPtr = std::make_shared<const Observation>(observation2D, mFocalLength);
    mSupportingPupils.push_back(newObservationPtr);
    mSupportingPupilsSize = mSupportingPupils.size();

    // Adding pupil center
    pupil_centers.conservativeResize(2,mSupportingPupilsSize);
    pupil_centers.col(mSupportingPupilsSize-1)(0) = observation2D->ellipse.center[0]+320.0;
    pupil_centers.col(mSupportingPupilsSize-1)(1) = observation2D->ellipse.center[1]+240.0;

    // Adding potential gaze vectors from unprojection of ellipse
    auto const gaze_vector_candidates = newObservationPtr->getUnprojectedCirclePair();
    gaze_vector_candidates_1.conservativeResize(3,mSupportingPupilsSize);
    gaze_vector_candidates_1.col(mSupportingPupilsSize-1) = std::move(gaze_vector_candidates.first.normal);
    gaze_vector_candidates_2.conservativeResize(3,mSupportingPupilsSize);
    gaze_vector_candidates_2.col(mSupportingPupilsSize-1) = std::move(gaze_vector_candidates.second.normal);

    // Return current number of stored observations
    return mSupportingPupilsSize;

}

EyeModel_v2::Sphere EyeModel_v2::findSphereCenter(bool use_ransac)
{
    using math::sq;

    Sphere sphere;

    if (mSupportingPupils.size()<2){

        return Sphere::Null;

    }

    const double eyeZ = 57.0; // This is an abitrary value, sphere will be scaled later

    std::vector<Line> pupilGazelinesProjected;

    for (const auto& pupil : mSupportingPupils)
    {

        pupilGazelinesProjected.push_back(pupil.mObservationPtr->getProjectedCircleGaze());

    }

    // Get eyeball center
    //
    // Find a least-squares 'intersection' (point nearest to all lines) of
    // the projected 2D gaze vectors. Then, unproject that circle onto a
    // point a fixed distance away.
    //
    // For robustness, use RANSAC to eliminate stray gaze lines
    //
    // (This has to be done here because it's used by the pupil circle
    // disambiguation)

    Vector2 eyeCenterProjected;
    bool validEye;

    if (use_ransac) {
        auto indices = fun::range_<std::vector<size_t>>(pupilGazelinesProjected.size());
        const int n = 2;
        double w = 0.3;
        double p = 0.9999;
        int k = ceil(log(1 - p) / log(1 - pow(w, n)));
        double epsilon = 10;
        auto error = [&](const Vector2 & point, const Line & line) {
            double dist = euclidean_distance(point, line);

            if (sq(dist) < sq(epsilon))
                return sq(dist);
            else
                return sq(epsilon);
        };
        auto bestInlierIndices = decltype(indices)();
        Vector2 bestEyeCenterProjected;
        double bestLineDistanceError = std::numeric_limits<double>::infinity();
        for (int i = 0; i < k; ++i) {
            auto indexSample = singleeyefitter::randomSubset(indices, n);
            auto sample = fun::map([&](size_t i) { return pupilGazelinesProjected[i]; }, indexSample);
            auto sampleCenterProjected = nearest_intersect(sample);
            auto indexInliers = fun::filter(
            [&](size_t i) { return euclidean_distance(sampleCenterProjected, pupilGazelinesProjected[i]) < epsilon; },
            indices);
            auto inliers = fun::map([&](size_t i) { return pupilGazelinesProjected[i]; }, indexInliers);

            if (inliers.size() <= w * pupilGazelinesProjected.size()) {
                continue;
            }

            auto inlierCenterProj = nearest_intersect(inliers);
            double lineDistanceError = fun::sum(
            [&](size_t i) { return error(inlierCenterProj, pupilGazelinesProjected[i]); },
            indices);

            if (lineDistanceError < bestLineDistanceError) {
                bestEyeCenterProjected = inlierCenterProj;
                bestLineDistanceError = lineDistanceError;
                bestInlierIndices = std::move(indexInliers);
            }
        }
        if (bestInlierIndices.size() > 0) {
            eyeCenterProjected = bestEyeCenterProjected;
            validEye = true;

        } else {
            validEye = false;
        }

    } else {

        eyeCenterProjected = nearest_intersect(pupilGazelinesProjected);
        validEye = true;

    }

    if (validEye) {

        sphere.center << eyeCenterProjected*eyeZ/mFocalLength, eyeZ;
        sphere.radius = 1;

        // Disambiguate pupil circles using projected eyeball center
        //
        // Assume that the gaze vector points away from the eye center, and
        // so projected gaze points away from projected eye center. Pick the
        // solution which satisfies this assumption

//        for (size_t i = 0; i < mSupportingPupils.size(); ++i) {
//
//            const auto& pupilPair = mSupportingPupils[i].mObservationPtr->getUnprojectedCirclePair();
//            const auto& line = mSupportingPupils[i].mObservationPtr->getProjectedCircleGaze();
//            const auto& originProjected = line.origin();
//            const auto& directionProjected = line.direction();
//
//            // Check if directionProjected going away from est eye center. If it is, then
//            // the first circle was correct. Otherwise, take the second one.
//            // The two normals will point in opposite directions, so only need
//            // to check one.
//            if ((originProjected - eyeCenterProjected).dot(directionProjected) >= 0) {
//                mSupportingPupils[i].mCircle =  pupilPair.first;
//
//            } else {
//                mSupportingPupils[i].mCircle = pupilPair.second;
//            }
//
//        }

    } else {

        // No inliers, so no eye
        sphere = Sphere::Null;
    }

    return sphere;

}

EyeModel_v2::Sphere EyeModel_v2::initialiseModel()
{

    Sphere sphere = findSphereCenter(false);

    if (sphere == Sphere::Null) {

        return sphere;

    }

    // Find pupil positions on eyeball to get radius
    //
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.
    // First estimate of pupil center, used only to get an estimate of eye radius
    double eyeRadiusAcc = 0;
    int eyeRadiusCount = 0;

    //for (const auto& pupil: mSupportingPupils) {
    std::vector<Pupil> initialisation_pupils = std::vector<Pupil>(mSupportingPupils.begin(),
                                                                  mSupportingPupils.begin()+std::min(mSupportingPupilsSize,100));
    Vector2 eyeCenterProjected;

    eyeCenterProjected[0] = sphere.center[0]*mFocalLength/57.0;
    eyeCenterProjected[1] = sphere.center[1]*mFocalLength/57.0;

    //sphere.center << eyeCenterProjected*eyeZ/mFocalLength, eyeZ;

    for (auto& pupil: initialisation_pupils) {

        const auto& pupilPair = pupil.mObservationPtr->getUnprojectedCirclePair();
        const auto& line = pupil.mObservationPtr->getProjectedCircleGaze();
        const auto& originProjected = line.origin();
        const auto& directionProjected = line.direction();

        // Check if directionProjected going away from est eye center. If it is, then
        // the first circle was correct. Otherwise, take the second one.
        // The two normals will point in opposite directions, so only need
        // to check one.
        if ((originProjected - eyeCenterProjected).dot(directionProjected) >= 0){
            pupil.mCircle =  pupilPair.first;
        }else{
            pupil.mCircle = pupilPair.second;
        }

        // Intersect the gaze from the eye center with the pupil circle
        // center projection line (with perfect estimates of gaze, eye
        // center and pupil circle center, these should intersect,
        // otherwise find the nearest point to both lines)
        Vector3 pupilCenter = nearest_intersect(Line3(sphere.center, pupil.mCircle.normal),
                                                Line3(mCameraCenter, pupil.mCircle.center.normalized()));
        auto distance = (pupilCenter-sphere.center).norm();
        eyeRadiusAcc += distance;
        ++eyeRadiusCount;
    }

    // Set the eye radius as the mean distance from pupil centers to eye center
    sphere.radius = eyeRadiusAcc / eyeRadiusCount;

    // Second estimate of pupil radius, used to get position of pupil on eye
//    for (auto& pupil : mSupportingPupils) {
//        initialiseSingleObservation(sphere, pupil);
//    }

    // Scale eye to anthropomorphic average radius of 10.39mm
    auto scale = 10.39 / sphere.radius;
    sphere.radius = 10.39;
    sphere.center *= scale;

//    for (auto& pupil : mSupportingPupils) {
//        pupil.mParams.radius *= scale;
//        pupil.mCircle = circleFromParams(sphere, pupil.mParams);
//    }

    mInitialSphere = sphere;

    return mInitialSphere;

}

EyeModel_v2::Sphere EyeModel_v2::optimizeModel()
{

    initialiseModel();

    Eigen::Matrix<double, 3, 1> x = mInitialSphere.center;

    ceres::Problem problem;
    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GazeAngleLossFunctionVectorized<double>, ceres::DYNAMIC, 3>(
                new GazeAngleLossFunctionVectorized<double>(pupil_centers,
                                                            gaze_vector_candidates_1,
                                                            gaze_vector_candidates_2,
                                                            10.39,
                                                            mFocalLength,
                                                            0.05), //Todo: Needs to get CauchLossScale as parameter
                mSupportingPupilsSize
            ),
            NULL,
            &x[0]
    );

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    mSphere.center = x;
    mSphere.radius = 10.39;

    return mSphere;

}

const Circle& EyeModel_v2::selectUnprojectedCircle(const Sphere& sphere, const std::pair<const Circle, const Circle>& circles) const
{
    const Vector3& c = circles.first.center;
    const Vector3& v = circles.first.normal;
    Vector2 centerProjected = project(c, mFocalLength);
    Vector2 directionProjected = project(v + c, mFocalLength) - centerProjected;
    directionProjected.normalize();
    Vector2 eyeCenterProjected = project(sphere.center, mFocalLength);

    if ((centerProjected-eyeCenterProjected).dot(directionProjected)>=0){

        return circles.first;

    }else{

       return circles.second;

    }

}

Circle EyeModel_v2::getIntersectedCircle(const Sphere& sphere, const Circle& circle) const
{
    // Ignore the circle normal, and intersect the circle
    // center projection line with the sphere
    std::pair<Vector3,Vector3> pupil_center_sphere_intersect;
    bool didIntersect =  intersect(Line3(mCameraCenter, circle.center.normalized()), sphere, pupil_center_sphere_intersect);

    if(didIntersect){

        auto new_pupil_center = pupil_center_sphere_intersect.first;
        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = circle.radius / circle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();
        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - sphere.center;
        double r = center_to_pupil.norm();
        double theta = acos(center_to_pupil[1] / r);
        double psi = atan2(center_to_pupil[2], center_to_pupil[0]);
        double radius = new_pupil_radius;
        // Update pupil circle to match parameters
        auto pupilParams = PupilParams(theta, psi, radius);
        return  circleFromParams(sphere,  pupilParams);

    } else {
        return Circle::Null;
    }

}

// Utilities

void EyeModel_v2::prepareObservation(std::shared_ptr<Detector2DResult>& observation2D) const
{

            int image_height = observation2D->image_height;
            int image_width = observation2D->image_width;
            int image_height_half = image_height / 2.0;
            int image_width_half = image_width / 2.0;

            Ellipse& ellipse = observation2D->ellipse;
            ellipse.center[0] -= image_width_half;
            ellipse.center[1] = image_height_half - ellipse.center[1];
            ellipse.angle = -ellipse.angle; //take y axis flip into account

            // Observation edge data are relative to their ROI
            cv::Rect roi = observation2D->current_roi;

            // put the edges int or coordinate system
            // edges are needed for every optimisation step
            for (cv::Point& p : observation2D->final_edges){
                p += roi.tl();
                p.x -= image_width_half;
                p.y  = image_height_half - p.y;
            }

}

Circle EyeModel_v2::circleFromParams(const Sphere& eye, const PupilParams& params) const
{
    if (params.radius == 0){

        return Circle::Null;

    }

    Vector3 radial = math::sph2cart<double>(double(1.0), params.theta, params.psi);

    return Circle(eye.center + eye.radius * radial,
                  radial,
                  params.radius);
}

void EyeModel_v2::reset()
{
        mSphere = Sphere::Null;
        mInitialSphere = Sphere::Null;
        mBirthTimestamp = -1;
        pupil_centers.resize(0,2);
        gaze_vector_candidates_1.resize(0,3);
        gaze_vector_candidates_2.resize(0,3);
        mSupportingPupils.clear();
        mSupportingPupilsSize = 0;

}

} // singleeyefitter
