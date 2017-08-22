

#include "EyeModel.h"

#include <algorithm>
#include <future>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

#include "EllipseDistanceApproxCalculator.h"
#include "EllipseDistanceResidualFunction.h"
#include "RefractionResidualFunction.h"

#include "CircleDeviationVariance3D.h"
#include "CircleEvaluation3D.h"
#include "CircleGoodness3D.h"

#include "utils.h"
#include "math/intersect.h"
#include "projection.h"
#include "fun.h"

#include "mathHelper.h"
#include "math/distance.h"




namespace singleeyefitter {



// EyeModel::EyeModel(EyeModel&& that) :
//     mInitialUncheckedPupils(that.mInitialUncheckedPupils), mFocalLength(that.mFocalLength), mCameraCenter(that.mCameraCenter),
//     mTotalBins(that.mTotalBins), mBinResolution(that.mBinResolution), mTimestamp(that.mTimestamp ),
//     mModelID(that.mModelID)
// {
//     std::lock_guard<std::mutex> lock(that.mModelMutex);
//     mSupportingPupils = std::move(that.mSupportingPupils);
//     mSupportingPupilsToAdd = std::move(that.mSupportingPupilsToAdd);
//     mSphere = std::move(that.mSphere);
//     mInitialSphere = std::move(that.mInitialSphere);
//     mSpatialBins = std::move(that.mSpatialBins);
//     mBinPositions = std::move(that.mBinPositions);
//     mFit = std::move(that.mFit);
//     mPerformance = std::move(that.mPerformance);
//     mMaturity = std::move(that.mMaturity);
//     mModelSupports = std::move(that.mModelSupports);
//     mSupportingPupilSize = mSupportingPupils.size();
//     std::cout << "MOVE EYE MODEL" << std::endl;
// }

// EyeModel& EyeModel::operator=(EyeModel&& that){

//     std::lock_guard<std::mutex> lock(that.mModelMutex);
//     mSupportingPupils = std::move(that.mSupportingPupils);
//     mSupportingPupilsToAdd = std::move(that.mSupportingPupilsToAdd);
//     mSphere = std::move(that.mSphere);
//     mInitialSphere = std::move(that.mInitialSphere);
//     mSpatialBins = std::move(that.mSpatialBins);
//     mBinPositions = std::move(that.mBinPositions);
//     mFit = std::move(that.mFit);
//     mPerformance = std::move(that.mPerformance);
//     mMaturity = std::move(that.mMaturity);
//     mModelSupports = std::move(that.mModelSupports);
//     mSupportingPupilSize = mSupportingPupils.size();
//     std::cout << "MOVE ASSGINE EYE MODEL" << std::endl;
//     return *this;
// }

EyeModel::EyeModel( int modelId, double timestamp,  double focalLength, Vector3 cameraCenter, int initialUncheckedPupils, double binResolution  ):
    mModelID(modelId),
    mBirthTimestamp(timestamp),
    mFocalLength(std::move(focalLength)),
    mCameraCenter(std::move(cameraCenter)),
    mInitialUncheckedPupils(initialUncheckedPupils),
    mTotalBins(std::pow(std::floor(1.0/binResolution), 2 ) * 4 ),
    mBinResolution(binResolution),
    mSolverFit(0),
    mPerformance(30),
    mPerformanceGradient(0),
    mLastPerformanceCalculationTime(),
    mPerformanceWindowSize(3.0),
    mEdgeNumber(20),
    mEyeballRadius(12.0),
    mCorneaRadius(7.5),
    mIrisRadius(6.0)
    {};

EyeModel::~EyeModel(){

    //wait for thread to finish before we dealloc
    if( mWorker.joinable() )
        mWorker.join();
}


std::pair<Circle,ConfidenceValue> EyeModel::presentObservation(const ObservationPtr newObservationPtr, double averageFramerate )
{

    if (mBirthTimestamp == -1){
        mBirthTimestamp = newObservationPtr->getObservation2D()->timestamp;
    }

    Circle circle;
    PupilParams currentPupilParams;
    bool shouldAddObservation = false;
    double confidence2D = newObservationPtr->getObservation2D()->confidence;
    ConfidenceValue oberservation_fit = ConfidenceValue(0,1);

    // unlock when done
    mModelMutex.lock(); // needed for mSphere and mSupportingPupilSize

    //Check for properties if it's a candidate we can use
    if (mSphere != Sphere::Null && (mSupportingPupilSize + mSupportingPupilsToAdd.size()) >= mInitialUncheckedPupils ) {

        // select the right circle depending on the current model
        const Circle& unprojectedCircle = selectUnprojectedCircle(mSphere, newObservationPtr->getUnprojectedCirclePair() );

        circle = getInitialCircle(mSphere, unprojectedCircle);
        std::pair<PupilParams, double> refraction_result = getRefractedCircle(mSphere, circle, newObservationPtr);
        currentPupilParams = refraction_result.first;
        circle = circleFromParams(mSphere, currentPupilParams);
        double cost = refraction_result.second;

        // initialised circle. circle parameters addapted to our current eye model
        circle = getIntersectedCircle(mSphere, unprojectedCircle);

        if (unprojectedCircle != Circle::Null && circle != Circle::Null) {  // initialise failed
            oberservation_fit = calculateModelOberservationFit(unprojectedCircle, circle , confidence2D);
            updatePerformance( oberservation_fit, averageFramerate);
        }

        if (circle == Circle::Null){
            circle = unprojectedCircle; // at least return the unprojected circle
        }

        //check first if the observations is strong enough to build the eye model ontop of it
        // the confidence is above 0.99 only if we have a strong prior.
        // also binchecking
        if (confidence2D > 0.99 && isSpatialRelevant(unprojectedCircle)) {
            shouldAddObservation = true;
        } else {
            //std::cout << " spatial check failed"  << std::endl;
        }

    }
    else { // no valid sphere yet

        shouldAddObservation = true;
        currentPupilParams = PupilParams(0.0,0.0,0.0);

    }

    mModelMutex.unlock();

    if (shouldAddObservation) {
        //if the observation passed all tests we can add it
        mSupportingPupilsToAdd.emplace_back( newObservationPtr, currentPupilParams);
    }

    using namespace std::chrono;
    Clock::time_point now( Clock::now() );
    seconds pastSecondsRefinement = duration_cast<seconds>(now - mLastModelRefinementTime);
    int amountNewObservations = mSupportingPupilsToAdd.size();

    if( amountNewObservations > 1 &&  pastSecondsRefinement.count() + amountNewObservations > 10   ){

            if(tryTransferNewObservations()) {

                auto work = [&](){

                    std::lock_guard<std::mutex> lockPupil(mPupilMutex);
                    auto sphere  = initialiseModel();
                    auto sphere2 = sphere;
                    double fit = refineWithEdges(sphere);

                    {
                        std::lock_guard<std::mutex> lockModel(mModelMutex);
                        mInitialSphere = sphere2;
                        mSphere = sphere;
                        mSolverFit = fit;
                    }

                };

                // needed in order to assign a new thread
                if(mWorker.joinable()){

                    mWorker.join(); // we should never wait here because tryTransferNewObservations is false if the work isn't finished

                }

                mLastModelRefinementTime =  Clock::now() ;
                mWorker = std::thread(work);
                //work();
            }
    }

    return {circle, oberservation_fit};
}

EyeModel::Sphere EyeModel::findSphereCenter( bool use_ransac /*= true*/)
{
    using math::sq;

    Sphere sphere;

    if (mSupportingPupils.size() < 2) {
        return Sphere::Null;
    }
    const double eyeZ = 57; // could be any value

    // should we save them some where else ?
    std::vector<Line> pupilGazelinesProjected;
    for (const auto& pupil : mSupportingPupils) {
        pupilGazelinesProjected.push_back( pupil.mObservationPtr->getProjectedCircleGaze() );
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

    if ( use_ransac ) {
        auto indices = fun::range_<std::vector<size_t>>(pupilGazelinesProjected.size());
        const int n = 2;
        double w = 0.3;
        double p = 0.9999;
        int k = ceil(log(1 - p) / log(1 - pow(w, n)));
        double epsilon = 10;
        // auto huber_error = [&](const Vector2 & point, const Line & line) {
        //     double dist = euclidean_distance(point, line);

        //     if (sq(dist) < sq(epsilon))
        //         return sq(dist) / 2;
        //     else
        //         return epsilon * (abs(dist) - epsilon / 2);
        // };
        auto error = [&](const Vector2 & point, const Line & line) {
            double dist = euclidean_distance(point, line);

            if (sq(dist) < sq(epsilon))
                return sq(dist);
            else
                return sq(epsilon);
        };
        auto bestInlierIndices = decltype(indices)();
        Vector2 bestEyeCenterProjected;// = nearest_intersect(pupilGazelinesProjected);
        double bestLineDistanceError = std::numeric_limits<double>::infinity();// = fun::sum(LAMBDA(const Line& line)(error(bestEyeCenterProjected,line)), pupilGazelinesProjected);

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

        // std::cout << "Inliers: " << bestInlierIndices.size()
        //     << " (" << (100.0*bestInlierIndices.size() / pupilGazelinesProjected.size()) << "%)"
        //     << " = " << bestLineDistanceError
        //     << std::endl;

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
        sphere.center << eyeCenterProjected* eyeZ / mFocalLength,
                   eyeZ;
        sphere.radius = 1;

        // Disambiguate pupil circles using projected eyeball center
        //
        // Assume that the gaze vector points away from the eye center, and
        // so projected gaze points away from projected eye center. Pick the
        // solution which satisfies this assumption

        for (size_t i = 0; i < mSupportingPupils.size(); ++i) {
            const auto& pupilPair = mSupportingPupils[i].mObservationPtr->getUnprojectedCirclePair();
            const auto& line = mSupportingPupils[i].mObservationPtr->getProjectedCircleGaze();
            const auto& originProjected = line.origin();
            const auto& directionProjected = line.direction();

            // Check if directionProjected going away from est eye center. If it is, then
            // the first circle was correct. Otherwise, take the second one.
            // The two normals will point in opposite directions, so only need
            // to check one.
            if ((originProjected - eyeCenterProjected).dot(directionProjected) >= 0) {
                mSupportingPupils[i].mCircle =  pupilPair.first;

            } else {
                mSupportingPupils[i].mCircle = pupilPair.second;
            }

            // calculate the center variance of the projected gaze vectors to the current eye center
          //  center_distance_variance += euclidean_distance_squared( eye.center, Line3(pupils[i].circle.center, pupils[i].circle.normal ) );

        }
        //center_distance_variance /= pupils.size();
        //std::cout << "center distance variance " << center_distance_variance << std::endl;

    } else {
        // No inliers, so no eye
        sphere = Sphere::Null;
    }

    return sphere;

}

EyeModel::Sphere EyeModel::initialiseModel()
{


    Sphere sphere = findSphereCenter();

    if (sphere == Sphere::Null) {
        return sphere;
    }
    //std::cout << "init model" << std::endl;

    // Find pupil positions on eyeball to get radius
    //
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.
    // First estimate of pupil center, used only to get an estimate of eye radius
    double eyeRadiusAcc = 0;
    int eyeRadiusCount = 0;

    for (const auto& pupil : mSupportingPupils) {

        // Intersect the gaze from the eye center with the pupil circle
        // center projection line (with perfect estimates of gaze, eye
        // center and pupil circle center, these should intersect,
        // otherwise find the nearest point to both lines)
        Vector3 pupilCenter = nearest_intersect(Line3(sphere.center, pupil.mCircle.normal),
                               Line3(mCameraCenter, pupil.mCircle.center.normalized()));
        auto distance = (pupilCenter - sphere.center).norm();
        eyeRadiusAcc += distance;
        ++eyeRadiusCount;
    }

    // Set the eye radius as the mean distance from pupil centers to eye center
    sphere.radius = eyeRadiusAcc / eyeRadiusCount;

    // Second estimate of pupil radius, used to get position of pupil on eye

    //TODO do we really need this if we don't do refinement ?????
    for ( auto& pupil : mSupportingPupils) {
        initialiseSingleObservation(sphere, pupil);
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / sphere.radius;
    sphere.radius = 12.0;
    sphere.center *= scale;
    for ( auto& pupil : mSupportingPupils) {
        pupil.mParams.radius *= scale;
        pupil.mCircle = circleFromParams(sphere, pupil.mParams);
    }

    return sphere;

}

double EyeModel::refineWithEdges(Sphere& sphere)
{

    int N = 5 + 3 * mSupportingPupils.size();
    std::vector<double> x;
    for (int i=0;i<N;i++){
        x.push_back(0);
    }

    // SETTING INITIAL PARAMETERS

    // EYE
    x[0] = sphere.center[0];
    x[1] = sphere.center[1];
    x[2] = sphere.center[2];
    x[3] = mCorneaRadius;
    x[4] = mIrisRadius;

    // PUPILS
    for (int i = 0; i < mSupportingPupils.size(); ++i) {
        const PupilParams& pupilParams = mSupportingPupils[i].mParams;
        x[5 + 3 * i + 0] = pupilParams.theta;
        x[5 + 3 * i + 1] = pupilParams.psi;
        x[5 + 3 * i + 2] = pupilParams.radius;
    }

    ceres::Problem problem;

    for (int i = 0; i < mSupportingPupils.size(); ++i) {

        const auto& pupilInliers = mSupportingPupils[i].mObservationPtr->getObservation2D()->final_edges;
        const Ellipse& ellipse = mSupportingPupils[i].mObservationPtr->getObservation2D()->ellipse;
        const cv::Point ellipse_center(ellipse.center[0], ellipse.center[1]);

        int N_;
        if (mEdgeNumber==-1){
            N_ = pupilInliers.size();
        }else{
            N_ = mEdgeNumber < pupilInliers.size() ? mEdgeNumber : pupilInliers.size();
        }

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RefractionResidualFunctionEyeParams<double>, ceres::DYNAMIC, 3, 2, 3>(
            new RefractionResidualFunctionEyeParams<double>(pupilInliers, mEyeballRadius, mFocalLength, i, ellipse_center, 1.0, 0.0, N_),
            N_+1),
            NULL, &x[0], &x[3], &x[5 + 3 * i]);

    }

    // SETTING BOUNDS

    // Z-POSITION OF SPHERE
    problem.SetParameterLowerBound(&x[0],2,15.0);
    problem.SetParameterUpperBound(&x[0],2,50.0);

    // PUPIL RADIUS
    for (int i=0;i<mSupportingPupils.size();i++){
       problem.SetParameterLowerBound(&x[5+i*3], 2, 1.0);
       problem.SetParameterUpperBound(&x[5+i*3], 2, 4.0);
    }

    // SETTING CERES OPTIONS
    ceres::Solver::Options options;
    options.logging_type = ceres::SILENT;
    options.linear_solver_type = ceres::DENSE_QR;
    options.gradient_tolerance = 1e-14;
    options.function_tolerance = 1e-14;
    options.parameter_tolerance = 1e-14;
    options.minimizer_progress_to_stdout = false;

    // SUMMARY SETUP
    ceres::Solver::Summary summary;

    //SETTING EYE PARAMTERS CONSTANT
    problem.SetParameterBlockConstant(&x[3]);

    // RUNNING SOLVER ->  ROUGH OPTIMIZATION SPHERE
    options.max_num_iterations = 5;
    for (int i = 0; i < mSupportingPupils.size(); ++i) {
        problem.SetParameterBlockConstant(&x[5 + 3 * i]);
    }
    ceres::Solve(options, &problem, &summary);

//    //ADJUST RADII
//    double distance_initial = mInitialSphere.center.norm();
//    double distance_current = sqrt(parameter_vector[0]*parameter_vector[0]+parameter_vector[1]*parameter_vector[1]+parameter_vector[2]*parameter_vector[2]);
//    for (int i = N2; i < mSupportingPupils.size(); ++i){
//        parameter_vector[5 + 3 * i + 2] *= distance_current/distance_initial;
//    }

    // RUNNING SOLVER ->  ROUGH OPTIMIZATION PUPILS
    options.max_num_iterations = 5;
    problem.SetParameterBlockConstant(&x[0]);
    for (int i = 0; i < mSupportingPupils.size(); ++i){
        problem.SetParameterBlockVariable(&x[5 + 3 * i]);
    }
    ceres::Solve(options, &problem, &summary);

    // RUNNING SOLVER ->  FINAL TUNING
    options.max_num_iterations = 20;
    problem.SetParameterBlockVariable(&x[0]);
    ceres::Solve(options, &problem, &summary);

    // UPDATING MODEL PARAMETERS FROM OPTIMIZATION RESULT
    sphere.center[0] = x[0];
    sphere.center[1] = x[1];
    sphere.center[2] = x[2];
    sphere.radius = sqrt(pow(mEyeballRadius, 2)-pow(mIrisRadius, 2));

//    // CHECKING RESIDUALS
//    double cost;
//    mCostPerBlock = std::vector<double>();
//    ceres::Problem::EvaluateOptions options_eval;
//    std::vector<ceres::ResidualBlockId> single_id;
//    single_id.push_back(residualblock_vector[0]);
//
//    for (auto& id: residualblock_vector){
//
//        single_id[0] = id;
//        options_eval.residual_blocks =  single_id;
//        problem.Evaluate(options_eval, &cost, &residuals, NULL, NULL);
//        mCostPerBlock.push_back(cost);
//
//    }

    return summary.final_cost;

}


bool EyeModel::tryTransferNewObservations()
{
    bool ownPupil = mPupilMutex.try_lock();
    if( ownPupil ){
        for( auto& pupil : mSupportingPupilsToAdd){
            mSupportingPupils.push_back( std::move(pupil) );
        }
        mSupportingPupilsToAdd.clear();
        mPupilMutex.unlock();
        std::lock_guard<std::mutex> lockModel(mModelMutex);
        mSupportingPupilSize = mSupportingPupils.size();
        return true;
    }else{
        return false;
    }

}

bool EyeModel::isSpatialRelevant(const Circle& circle)
{

 /* In order to check if new observations are unique (not in the same area as previous one ),
     the position on the sphere (only x,y coords) are binned  (spatial binning) an inserted into the right bin.
     !! This is not a correct method to check if they are uniformly distibuted, because if x,y are uniformly distibuted
     it doesn't mean points on the spehre are uniformly distibuted.
     To uniformly distribute points on a sphere you have to check if the area is equal on the sphere of two bins.
     We just look on half of the sphere, it's like projecting a checkboard grid on the sphere, thus the bins are more dense in the projection center,
     and less dense further back.
     Still it gives good results an works for our purpose
    */
    Vector3 pupilNormal =  circle.normal; // the same as a vector from unit sphere center to the pupil center

    // calculate bin
    // values go from -1 to 1
    double x = pupilNormal.x();
    double y = pupilNormal.y();
    x = math::round(x , mBinResolution);
    y = math::round(y , mBinResolution);

    Vector2 bin(x, y);
    auto search = mSpatialBins.find(bin);

    if (search == mSpatialBins.end() || search->second == false) {

        // there is no bin at this coord or it is empty
        // so add one
        mSpatialBins.emplace(bin, true);
        double z = std::copysign(std::sqrt(1.0 - x * x - y * y),  pupilNormal.z());
        Vector3 binPositions3D(x , y, z); // for visualization
        mBinPositions.push_back(std::move(binPositions3D));
        return true;
    }

    return false;

}

void EyeModel::initialiseSingleObservation( const Sphere& sphere, Pupil& pupil) const
{
    // Ignore the circle normal, and intersect the circle
    // center projection line with the sphere

    std::pair<Vector3,Vector3> pupil_center_sphere_intersect;
    bool didIntersect =  intersect(Line3(mCameraCenter, pupil.mCircle.center.normalized()), sphere, pupil_center_sphere_intersect);

    if(didIntersect){

        auto new_pupil_center = pupil_center_sphere_intersect.first;
        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = pupil.mCircle.radius / pupil.mCircle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();
        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - sphere.center;
        double r = center_to_pupil.norm();
        pupil.mParams.theta = acos(center_to_pupil[1] / r);
        pupil.mParams.psi = atan2(center_to_pupil[2], center_to_pupil[0]);
        pupil.mParams.radius = new_pupil_radius;
        // Update pupil circle to match parameters
        pupil.mCircle = circleFromParams(sphere,  pupil.mParams );


    } else {

        // pupil.mCircle =  Circle::Null;
        // pupil.mParams = PupilParams();
        auto pupil_radius_at_1 = pupil.mCircle.radius / pupil.mCircle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * sphere.center.z();
        pupil.mParams.radius = new_pupil_radius;
        pupil.mParams.theta = acos(pupil.mCircle.normal[1] / sphere.radius);
        pupil.mParams.psi = atan2(pupil.mCircle.normal[2], pupil.mCircle.normal[0]);
        // Update pupil circle to match parameters
        pupil.mCircle = circleFromParams(sphere,  pupil.mParams );

    }


}


// PERFORMANCE RELATED

ConfidenceValue EyeModel::calculateModelOberservationFit(const Circle&  unprojectedCircle, const Circle& initialisedCircle, double confidence2D) const
{

    // the angle between the unprojected and the initialised circle normal tells us how good the current observation supports our current model
    // if our model is good these normals should align.
    const auto& n1 = unprojectedCircle.normal;
    const auto& n2 = initialisedCircle.normal;
    ConfidenceValue oberservationFit;
    oberservationFit.value = n1.dot(n2);

    // if the 2d pupil is almost a circle the unprojection gets inaccurate, thus the normal doesn't align well with the initialised circle
    // this is the case when looking directly into the camera.
    // we take this into account be calculation a confidence which depends on the angle between the normal and the direction from the sphere to the camera
    const Vector3 sphereToCameraDirection = (mCameraCenter - mSphere.center).normalized();
    const double eccentricity = sphereToCameraDirection.dot(initialisedCircle.normal);
    //std::cout << "inaccuracy: " <<  inaccuracy << std::endl;

    // then we calculate a how much we usefullness we give the oberservationFit value by merging the 2d confidence with eccentriciy.
    // we do this using a function with parameters that are tweaked through experimentation.
    // a plot of the fn can be found here:
    // http://www.livephysics.com/tools/mathematical-tools/online-3-d-function-grapher/?xmin=0&xmax=1&ymin=0&ymax=1&zmin=Auto&zmax=Auto&f=x%5E10%2A%281-y%5E20%29
    oberservationFit.confidence =  (1-pow(eccentricity,20)) * pow(confidence2D,15);

    return oberservationFit;
}

void EyeModel::updatePerformance( const ConfidenceValue& performance_datum, double averageFramerate )
{

    // dont add values with 0.0 confidence.
    if( performance_datum.value <= 0.0 )
        return;

    const double previousPerformance = mPerformance.getAverage();

    // whenever there is a change in framerate bigger than 1, change the window size
    // window size linearly depends on the framerate
    // the average frame rate changes slowly to compensate onetime big changes
    if( std::abs(averageFramerate  - mPerformance.getWindowSize()/mPerformanceWindowSize) > 1 ){
        int newWindowSize = std::round(  averageFramerate * mPerformanceWindowSize );
        mPerformance.changeWindowSize(newWindowSize);
    }

    mPerformance.addValue(performance_datum.value , performance_datum.confidence); // weighted average

    using namespace std::chrono;

    Clock::time_point now( Clock::now() );
    duration<double, std::milli> deltaTimeMs = now - mLastPerformanceCalculationTime;
    // calculate performance gradient (backward difference )
    mPerformanceGradient =  (mPerformance.getAverage() - previousPerformance) / deltaTimeMs.count();
    mLastPerformanceCalculationTime =  now;
}

double EyeModel::calculateModelFit(const Circle&  unprojectedCircle, const Circle& optimizedCircle) const
{

    // the angle between the unprojected and the initialised circle normal tells us how good the current observation supports our current model
    // if our model is good and the camera didn't change perspective or so, these normals should align pretty well
    const auto& n1 = unprojectedCircle.normal;
    const auto& n2 = optimizedCircle.normal;
    const double normalsAngle = n1.dot(n2);
    return normalsAngle;
}


// UTILITY FUNCTION

const Circle& EyeModel::selectUnprojectedCircle( const Sphere& sphere,  const std::pair<const Circle, const Circle>& circles) const
{
    const Vector3& c = circles.first.center;
    const Vector3& v = circles.first.normal;
    Vector2 centerProjected = project(c, mFocalLength);
    Vector2 directionProjected = project(v + c, mFocalLength) - centerProjected;
    directionProjected.normalize();
    Vector2 eyeCenterProjected = project(sphere.center, mFocalLength);

    if ((centerProjected - eyeCenterProjected).dot(directionProjected) >= 0) {
        return circles.first;

    } else {
       return circles.second;
    }

}

Circle EyeModel::circleFromParams(const Sphere& eye, const PupilParams& params) const
{
    if (params.radius == 0)
        return Circle::Null;

    Vector3 radial = math::sph2cart<double>(double(1), params.theta, params.psi);
    return Circle(eye.center + eye.radius * radial,
                  radial,
                  params.radius);
}

Circle EyeModel::getIntersectedCircle( const Sphere& sphere, const Circle& circle) const
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

Circle EyeModel::getInitialCircle( const Sphere& sphere, const Circle& circle ) const
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

    } else {  //GET CLOSEST POINT ON SPHERE

        auto parallel_length = circle.center.normalized().dot(sphere.center-mCameraCenter);
        auto new_pupil_center = (parallel_length*circle.center.normalized()+mCameraCenter-sphere.center).normalized()+sphere.center;
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
        return circleFromParams(sphere,  pupilParams);

    }

}

std::pair<EyeModel::PupilParams, double> EyeModel::getRefractedCircle( const Sphere& sphere, const Circle& unrefracted_circle, const ObservationPtr observation ) const
{

      Eigen::Matrix<double, Eigen::Dynamic, 1> par;
      par = Eigen::Matrix<double, Eigen::Dynamic, 1>(8);
      Eigen::Matrix<double, 3, 1> pupil_radial = unrefracted_circle.center-sphere.center;
      Eigen::Matrix<double, 2, 1> pupil_spherical = math::cart2sph(pupil_radial[0], pupil_radial[1], pupil_radial[2]);

      par.segment<3>(0) = mSphere.center;
      par[3] = mCorneaRadius;
      par[4] = mIrisRadius;
      par[5] = pupil_spherical[0];
      par[6] = pupil_spherical[1];
      par[7] = unrefracted_circle.radius;

      if ( par(7,0) < 0.1 ) par(7,0) = 1.0;
      if ( par(7,0) > 4 ) par(7,0) = 4.0;

      const auto& pupilInliers = observation->getObservation2D()->final_edges;
      const Ellipse& ellipse =  observation->getObservation2D()->ellipse;
      cv::Point ellipse_center(ellipse.center[0], ellipse.center[1]);

      ceres::Problem problem;

      int N_;
      if (mEdgeNumber==-1){
          N_ = pupilInliers.size();
      }else{
          N_ = mEdgeNumber < pupilInliers.size() ? mEdgeNumber : pupilInliers.size();
      }

      problem.AddResidualBlock(new ceres::AutoDiffCostFunction < RefractionResidualFunctionEyeParams<double>, ceres::DYNAMIC, 3, 2, 3 > (
                               new RefractionResidualFunctionEyeParams<double>(pupilInliers, mEyeballRadius, mFocalLength, 1, ellipse_center, 1.0, 0.0, N_),
                               N_+1),
                               NULL, &par[0], &par[3], &par[5]);

      problem.SetParameterBlockConstant(&par[0]);
      problem.SetParameterBlockConstant(&par[3]);
      problem.SetParameterLowerBound(&par[5], 2, 1.0);
      problem.SetParameterUpperBound(&par[5], 2, 4.0);

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = false;
      options.function_tolerance = 1e-14;
      options.max_num_iterations = 20;

      ceres::Solver::Summary summary;

      ceres::Solve(options, &problem, &summary);

      PupilParams pupilParams = PupilParams(par(5,0), par(6,0), par(7,0));

      return  {pupilParams, summary.final_cost};

}


// GETTER

EyeModel::Sphere EyeModel::getSphere() const
{
    std::lock_guard<std::mutex> lockModel(mModelMutex);
    return mSphere;
};

EyeModel::Sphere EyeModel::getInitialSphere() const
{
    std::lock_guard<std::mutex> lockModel(mModelMutex);
    return mInitialSphere;
};

double EyeModel::getMaturity() const
{

    //Spatial variance
    // Our bins are just on half of the sphere and by observing different models, it turned out
    // that if a eighth of half the sphere is filled it gives a good maturity.
    // Thus we scale it that a the maturity will be 1 if a eighth is filled
    using std::floor;
    using std::pow;
    return  mSpatialBins.size()/(mTotalBins/8.0);
}

double EyeModel::getConfidence() const
{

    return fmin(1.,fmax(0.,fmod(mPerformance.getAverage(),0.99)*100));
}

double EyeModel::getPerformance() const
{

    return mPerformance.getAverage();
}

double EyeModel::getPerformanceGradient() const
{

    return mPerformanceGradient;
}

double EyeModel::getSolverFit() const
{

    return mSolverFit;
}


} // singleeyefitter
