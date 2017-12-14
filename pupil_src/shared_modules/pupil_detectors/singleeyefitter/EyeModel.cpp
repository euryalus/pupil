

#include "EyeModel.h"

#include <algorithm>
#include <future>

#include <Eigen/Dense>

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
#include <stdlib.h>

#include "mathHelper.h"
#include "math/distance.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <map>


namespace singleeyefitter {

EyeModel::EyeModel(int modelId, double timestamp,  double focalLength, Vector3 cameraCenter, int initialUncheckedPupils, double binResolution):
    mModelID(modelId),
    mCurrentSphere(Sphere::Null),
    mCurrentInitialSphere(Sphere::Null),
    mSphere(Sphere::Null),
    mBirthTimestamp(timestamp),
    mFocalLength(focalLength),
    mCameraCenter(std::move(cameraCenter)),
    mInitialUncheckedPupils(initialUncheckedPupils),
    mTotalBins(std::pow(std::floor(1.0/binResolution), 2 ) * 4 ),
    mBinResolution(binResolution),
    mSolverFit(0),
    mPerformance(30),
    mPerformanceGradient(0),
    mLastPerformanceCalculationTime(),
    mPerformanceWindowSize(3.01),
    mEdgeNumber(20),
    mStrikes(2),
    mCenterWeightInitial(4.0),
    mCenterWeightFinal(0.0),
    mResidualsAveragedFraction(0.8),
    mOutlierFactor(7.0),
    mStartRemoveNumber(25),
    mCauchyLossScale(0.001),
    mEyeballRadius(12.0),
    mCorneaRadius(7.5),
    mIrisRadius(6.0),
    mInitialCorneaRadius(7.5),
    mInitialIrisRadius(6.0)
    {

        mCurrentSphere = Sphere::Null;
        mCurrentInitialSphere = Sphere::Null;

        srand(30948);

        mIterationNumbers[0] = 10;
        mIterationNumbers[1] = 20;
        mIterationNumbers[2] = 20;
        mIterationNumbers[3] = 20;
        mIterationNumbers[4] = 500;

        mResult.cost = 0.0;
        mResult.number_of_pupils = 0;
        mResult.par_history = std::vector<std::vector<double>>();
        mResult.pupil_type_history = std::vector<std::vector<int>>();
        mResult.cost_history = std::vector<double>();
        mResult.edge_map = std::map<int,std::vector<std::vector<double>>>();
        mResult.circles = std::vector<Circle>();
        mResult.ellipses  = std::vector<Ellipse>();

    }

EyeModel::~EyeModel(){

    //wait for thread to finish before we dealloc
    if( mWorker.joinable() )
        mWorker.join();

}


// General
Detector3DResult EyeModel::predictAndUpdate(std::shared_ptr<Detector2DResult>& observation2D, const Detector3DProperties& props, bool debug)
{
    mDebug = debug;
    Detector3DResult result;

    int image_height = observation2D->image_height;
    int image_width = observation2D->image_width;
    int image_height_half = image_height / 2.0;
    int image_width_half = image_width / 2.0;

    // Put observation data into the right coordinate system
    Ellipse& ellipse = observation2D->ellipse;
    ellipse.center[0] -= image_width_half;
    ellipse.center[1] = image_height_half - ellipse.center[1];
    ellipse.angle = -ellipse.angle; //take y axis flip into account

    // Observation edge data are realtive to their ROI
    cv::Rect roi = observation2D->current_roi;

    // Put edges in our coordinate system, as they are needed for every optimisation step
    for (cv::Point& p : observation2D->final_edges) {
        p += roi.tl();
        p.x -= image_width_half;
        p.y = image_height_half - p.y;
    }

    auto observation3DPtr = std::make_shared<const Observation>(observation2D, mFocalLength);

    if (observation2D->confidence >= 0.05){
        auto circleAndCost = presentObservation(observation3DPtr, 30.0);
        result.circle = circleAndCost.first;
        result.cost= circleAndCost.second;
    }else{
        result.circle = Circle::Null;
    }

    // Prepare the results object
    result.confidence = observation2D->confidence;
    result.timestamp = observation2D->timestamp;

    // getSphere contains a mutex and blocks if optimisation is running
    // Thus we just wanna call it once and save the results
    mCurrentSphere = getSphere();
    mCurrentInitialSphere = getInitialSphere();

    result.sphere = mCurrentSphere;
    result.projectedSphere = project(mCurrentSphere, mFocalLength);

    if(result.circle != Circle::Null){
        result.circle.radius = 4.0;
        result.ellipse = Ellipse(project(result.circle, mFocalLength));
    }else{
        result.confidence = 0.0;
        result.ellipse = Ellipse::Null;
   }

    result.modelID = getModelID();
    result.modelBirthTimestamp = getBirthTimestamp();

    if (mDebug) {

        result.models.reserve(1);

        ModelDebugProperties props;

        result.RefractionResult = getRefractionResult();
        props.optimizedParameters = getOptimizedParameters();
        props.costPerPupil = getCostPerPupil();
        props.resFit = getResFit();
        props.sphere = getSphere();
        props.initialSphere = getInitialSphere();
        props.modelID = getModelID();
        props.birthTimestamp = getBirthTimestamp();
        result.models.push_back(std::move(props));

    }

    return result;

}

std::pair<Circle, double> EyeModel::presentObservation(const ObservationPtr newObservationPtr, double averageFramerate )
{

    if (mBirthTimestamp == -1){
        mBirthTimestamp = newObservationPtr->getObservation2D()->timestamp;
    }

    Circle circle = Circle::Null;
    double cost = 0;

    PupilParams currentPupilParams;
    bool shouldAddObservation = false;
    double confidence2D = newObservationPtr->getObservation2D()->confidence;
    //ConfidenceValue oberservation_fit = ConfidenceValue(0,1);

    // unlock when done
    mModelMutex.lock(); // needed for mSphere and mSupportingPupilSize

    //Check for properties if it's a candidate we can use
    if (mSphere != Sphere::Null && (mSupportingPupilSize + mSupportingPupilsToAdd.size()) >= mInitialUncheckedPupils ) {

        // select the right circle depending on the current model
        const Circle& unprojectedCircle = selectUnprojectedCircle(mSphere, newObservationPtr->getUnprojectedCirclePair() );

        // we are initiliasing by looking for the closest point on the sphere and scaling the radius appropriately
        circle = getInitialCircle(mSphere, unprojectedCircle);
        std::pair<PupilParams, double> refraction_result = predictApproximate(mSphere, circle, newObservationPtr);
        currentPupilParams = refraction_result.first;
        circle = circleFromParams(mSphere, currentPupilParams);
        cost = refraction_result.second;

        if (circle == Circle::Null){
            circle = unprojectedCircle; // at least return the unprojected circle
        }

        // check first if the observations is strong enough to build the eye model ontop of it
        // the confidence is above 0.99 only if we have a strong prior.
        // also binchecking

        // here we decide whether we take a pupil into account, in particular, we are less srict on the confidence, if the angle with respect
        // to the optical axis is large
        Eigen::Matrix<double, 3, 1> normal = math::sph2cart<double>( currentPupilParams.radius,
                                                                             currentPupilParams.theta,
                                                                             currentPupilParams.psi);
        normal.normalize();
        Eigen::Matrix<double, 3, 1> axis = -mSphere.center;
        axis.normalize();
        double angle = acos(normal.dot(axis));
        int bin_number = static_cast<int>(angle/(3.142/20.0));

        if (
//        confidence2D > 0.90

            (bin_number<7 &&
            ((mSupportingPupilSize>60 && (confidence2D > 0.99 && isSpatialRelevant(unprojectedCircle))) ||
            ((mSupportingPupilSize>30 && mSupportingPupilSize<=60) && (confidence2D > 0.97 && isSpatialRelevant(unprojectedCircle))) ||
            (mSupportingPupilSize<=30 && (confidence2D > 0.96 && isSpatialRelevant(unprojectedCircle))))) ||

            (bin_number>=7 &&
            ((mSupportingPupilSize>60 && (confidence2D > 0.99 && isSpatialRelevant(unprojectedCircle))) ||
            ((mSupportingPupilSize>30 && mSupportingPupilSize<=60) && (confidence2D > 0.96 && isSpatialRelevant(unprojectedCircle))) ||
            (mSupportingPupilSize<=30 && (confidence2D > 0.93 && isSpatialRelevant(unprojectedCircle)))))


            ){
            shouldAddObservation = true;
            }else{
            //std::cout << " spatial check failed"  << std::endl;
            }

    }
    else { // no valid sphere yet

        shouldAddObservation = true;
        currentPupilParams = PupilParams(0.0, 0.0, 0.0);

    }

    mModelMutex.unlock();

    if (shouldAddObservation) {
        //if the observation passed all tests we can add it
        mSupportingPupilsToAdd.emplace_back(newObservationPtr, currentPupilParams);
    }

    using namespace std::chrono;
    Clock::time_point now(Clock::now());
    seconds pastSecondsRefinement = duration_cast<seconds>(now - mLastModelRefinementTime);
    int amountNewObservations = mSupportingPupilsToAdd.size();

    if( amountNewObservations > 1 &&  pastSecondsRefinement.count() + amountNewObservations > 10   ){

            if(tryTransferNewObservations()) {

                auto work = [&](){

                    Sphere sphere;
                    Sphere sphere2;

                    std::lock_guard<std::mutex> lockPupil(mPupilMutex);
                    if (mSupportingPupilSize<500){
                        sphere  = initialiseModel();
                        sphere2 = sphere;
                    }else{
                        sphere = mInitialSphere;
                        sphere2 = mSphere;
                    }
                    double fit = refineWithEdgesRefraction(sphere);

                    // SCOPE FOR LOCK_GUARD
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

    return {circle, cost};
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
        if (pupil.ceres_toggle<mStrikes){
        pupilGazelinesProjected.push_back( pupil.mObservationPtr->getProjectedCircleGaze() );
        }
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

    }
    else {

        eyeCenterProjected = nearest_intersect(pupilGazelinesProjected);
        validEye = true;
    }

    if (validEye) {
        sphere.center << eyeCenterProjected* eyeZ / mFocalLength, eyeZ;
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

    }
    else {
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
        if (pupil.ceres_toggle<mStrikes){
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
    }

    // Set the eye radius as the mean distance from pupil centers to eye center
    sphere.radius = eyeRadiusAcc / eyeRadiusCount;

    for ( auto& pupil : mSupportingPupils) {
        if (pupil.ceres_toggle<mStrikes){
            initialiseSingleObservation(sphere, pupil);
        }
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / sphere.radius;
    sphere.radius = 12.0;
    sphere.center *= scale;
    for ( auto& pupil : mSupportingPupils) {
        if (pupil.ceres_toggle<mStrikes){
            pupil.mParams.radius *= scale;
            if (pupil.mParams.radius<1.0){
                    pupil.mParams.radius=1.001;
            }
            if (pupil.mParams.radius>4.0){
                    pupil.mParams.radius=3.999;
            }
            pupil.mCircle = circleFromParams(sphere, pupil.mParams);
            pupil.optimizedParams[0] = pupil.mParams.theta;
            pupil.optimizedParams[1] = pupil.mParams.psi;
            pupil.optimizedParams[2] = pupil.mParams.radius;
        }
    }

    // SET INITIAL EYE PARAMETERS
    eye_params[0] = sphere.center[0];
    eye_params[1] = sphere.center[1];
    eye_params[2] = sphere.center[2];
    eye_params[3] = mInitialCorneaRadius;
    eye_params[4] = mInitialIrisRadius;

    return sphere;

}

Detector3DResultRefraction EyeModel::optimizeRefraction(bool initialization_toggle)
{
        if (initialization_toggle){

            auto sphere  = initialiseModel();
            auto sphere2 = sphere;
            mInitialSphere = sphere2;
            double fit = refineWithEdgesRefraction(sphere);

            mSphere = sphere;

            mSolverFit = fit;

            mResult.initial_center[0] = mInitialSphere.center[0];
            mResult.initial_center[1] = mInitialSphere.center[1];
            mResult.initial_center[2] = mInitialSphere.center[2];
            mResult.optimized_center[0] = mSphere.center[0];
            mResult.optimized_center[1] = mSphere.center[1];
            mResult.optimized_center[2] = mSphere.center[2];

        }else{

            mInitialSphere = mSphere;
            auto sphere = mSphere;
            double fit = refineWithEdgesRefraction(sphere);

            mSphere = sphere;

            mSolverFit = fit;

            mResult.initial_center[0] = mInitialSphere.center[0];
            mResult.initial_center[1] = mInitialSphere.center[1];
            mResult.initial_center[2] = mInitialSphere.center[2];
            mResult.optimized_center[0] = mSphere.center[0];
            mResult.optimized_center[1] = mSphere.center[1];
            mResult.optimized_center[2] = mSphere.center[2];

        }

        return mResult;

}

double EyeModel::refineWithEdgesRefraction(Sphere& sphere)
{
    std::cout<<"HEllo"<<std::endl;

    ceres::Problem problem;

    // PARAMTERS FOR COSTFUNCTION
    double center_weight = mCenterWeightInitial;
    double * const center_weight_ptr = &center_weight;

    // SAVING INITIAL POSITION OF EYEBALL
    double initial[3];
    initial[0] = eye_params[0];
    initial[1] = eye_params[1];
    initial[2] = eye_params[2];

    if (initial[2]>60.0){eye_params[2]=59.9;}
    if (initial[2]<15.0){eye_params[2]=15.1;}

    // ADDING NEW PUPILS
    int N_;

    std::vector<Circle> temp_circles;
    std::vector<Ellipse> temp_ellipses;
    std::map<int, std::vector<std::vector<double>>> temp_edge_map;
    std::vector<double*> all_par_blocks;
    std::vector<Pupil*> used_pupils;

    int i=0;
    for (auto& pupil: mSupportingPupils){

        // ONLY ADD NEW PUPILS TO THE PROBLEM THAT HAVE NOT BEEN OUTLIERS FOR mStrikes TIMES
        if (pupil.ceres_toggle<mStrikes){

            used_pupils.push_back(&pupil);

            const auto& pupilInliers = pupil.mObservationPtr->getObservation2D()->final_edges;
            const Ellipse& ellipse = pupil.mObservationPtr->getObservation2D()->ellipse;
            const cv::Point ellipse_center(ellipse.center[0], ellipse.center[1]);

            if (mEdgeNumber==-1){
                N_ = pupilInliers.size();
            }else{
                N_ = mEdgeNumber < pupilInliers.size() ? mEdgeNumber : pupilInliers.size();
            }

            ceres::CostFunction * current_cost = new ceres::AutoDiffCostFunction<RefractionResidualFunction<double>, ceres::DYNAMIC, 1, 1, 1, 2, 3>(
            new RefractionResidualFunction<double>(pupilInliers, mEyeballRadius, mFocalLength, ellipse_center, center_weight_ptr, N_),
            N_+1);
            pupil.mResidualBlockId = problem.AddResidualBlock(current_cost, new ceres::CauchyLoss(mCauchyLossScale), &eye_params[0],  &eye_params[1],  &eye_params[2], &eye_params[3], &(pupil.optimizedParams[0]));
            all_par_blocks.push_back(pupil.optimizedParams);

            // NEEDED FOR DEBUGGING
            temp_circles.push_back(selectUnprojectedCircle(mSphere, pupil.mObservationPtr->getUnprojectedCirclePair()));
            temp_ellipses.push_back(ellipse);

            //SAVING EDGES FOR DEBUGGING
            temp_edge_map.insert(std::make_pair(i, std::vector<std::vector<double>>()));
            std::vector<double> v;
            for (int j = 0; j < pupilInliers.size(); ++j){
                v = {static_cast<double>(pupilInliers[j].x), static_cast<double>(pupilInliers[j].y)};
                temp_edge_map[i].push_back(v);
            }
            i++;

        }
    }


    //SAVE ALL RESIDUALBLOCK-IDS IN A VECTOR
    std::vector<ceres::ResidualBlockId> residualblock_vector;
    problem.GetResidualBlocks(&residualblock_vector);

    // SETTING BOUNDS - Z-POSITION OF SPHERE
    problem.SetParameterLowerBound(&eye_params[2], 0, 15.0);
    problem.SetParameterUpperBound(&eye_params[2], 0, 60.0);

    // SETTING BOUNDS - PUPIL RADII
    std::vector<double*> par_blocks;


    for (const auto& rb: residualblock_vector){
        problem.GetParameterBlocksForResidualBlock(rb, &par_blocks);
        problem.SetParameterLowerBound(par_blocks[4], 2, 0.7);
        problem.SetParameterUpperBound(par_blocks[4], 2, 5.0);
    }

    // SETTING CERES OPTIONS
    ceres::Problem::Options prob_options;
    prob_options.enable_fast_removal = true;

    ceres::Solver::Options options;
    options.logging_type = ceres::SILENT;
   	options.use_nonmonotonic_steps = false;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.use_inner_iterations = false;
    options.gradient_tolerance = 1e-22;
    options.function_tolerance = 1e-22;
    options.parameter_tolerance = 1e-22;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;

    // CALLBACK
    class MyCallback2: public ceres::IterationCallback {

    public:

            MyCallback2(std::vector<double*> par, double * const eye_pars, std::vector<Pupil*> UsedPupils) : par_callback(par), mEyePars(eye_pars), mUsedPupils(UsedPupils)  {}

            virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
            {

                cost_history.push_back(summary.cost);

                std::vector<double> current_par;
                current_par.push_back(mEyePars[0]);
                current_par.push_back(mEyePars[1]);
                current_par.push_back(mEyePars[2]);
                current_par.push_back(mEyePars[3]);
                current_par.push_back(mEyePars[4]);
                for (auto block: par_callback)
                {
                    current_par.push_back(block[0]);
                    current_par.push_back(block[1]);
                    current_par.push_back(block[2]);

                }
                par_history.push_back(current_par);


                std::vector<int> current_pupil_type;
                for (const auto pupil: mUsedPupils){
                   current_pupil_type.push_back(pupil->ceres_toggle);
                }
                pupil_type_history.push_back(current_pupil_type);

                return ceres::SOLVER_CONTINUE;

            }

            std::vector<std::vector<double>> get_par_history(){
                return par_history;
            }

            std::vector<double> get_cost_history(){
                return cost_history;
            }

            std::vector<std::vector<int>> get_pupil_type_history(){
                return pupil_type_history;
            }

    private:

            std::vector<double*> par_callback;
            double * const mEyePars;
            std::vector<std::vector<double>> par_history;
            std::vector<std::vector<int>> pupil_type_history;
            std::vector<double> cost_history;
            std::vector<Pupil*> mUsedPupils;

    };

    // Needed for debugging
    MyCallback2 callback = MyCallback2(all_par_blocks, eye_params, used_pupils);
    options.callbacks.push_back(&callback);
    options.update_state_every_iteration = true;

    // SUMMARY SETUP
    ceres::Solver::Summary summary;

    //SETTING EYE GEOMETRY CONSTANT
    problem.SetParameterBlockConstant(&eye_params[2]);
    problem.SetParameterBlockConstant(&eye_params[3]);

    //SAVING INITIAL POSITION
    Eigen::Matrix<double,3,1> initial_pos{eye_params[0], eye_params[1], eye_params[2]};

    // RUNNING SOLVER -> ROUGH OPTIMIZATION SPHERE
    center_weight = mCenterWeightInitial;
    options.max_num_iterations =  mIterationNumbers[0];
    for (const auto rb: residualblock_vector){
        par_blocks = std::vector<double*>();
        problem.GetParameterBlocksForResidualBlock(rb, &par_blocks);
        problem.SetParameterBlockConstant(par_blocks[4]);
    }
    ceres::Solve(options, &problem, &summary);

    // RUNNING SOLVER ->  ROUGH OPTIMIZATION PUPILS
    center_weight = mCenterWeightInitial;
    options.max_num_iterations =  mIterationNumbers[1];
    problem.SetParameterBlockConstant(&eye_params[0]);
    problem.SetParameterBlockConstant(&eye_params[1]);
    problem.SetParameterBlockConstant(&eye_params[2]);
    for (const auto rb: residualblock_vector){
        par_blocks = std::vector<double*>();
        problem.GetParameterBlocksForResidualBlock(rb, &par_blocks);
        problem.SetParameterBlockVariable(par_blocks[4]);
    }
    ceres::Solve(options, &problem, &summary);

    // RUNNING SOLVER -> ALL PARAMS FREE
    center_weight = mCenterWeightInitial;
    options.max_num_iterations =  mIterationNumbers[2];
    problem.SetParameterBlockVariable(&eye_params[0]);
    problem.SetParameterBlockVariable(&eye_params[1]);
    problem.SetParameterBlockVariable(&eye_params[2]); //HERE WE CAN FIX THE z POSITION!
    ceres::Solve(options, &problem, &summary);

    // RUNNING SOLVER -> CENTER_WEIGHT TO ZERO
    center_weight = mCenterWeightFinal;
    options.max_num_iterations =  mIterationNumbers[3];
    ceres::Solve(options, &problem, &summary);

    // REMEMBER NUMBER OF INITIAL PUPILS IN OPTIMIZATION
    int first_round_number_residual_blocks = problem.NumResidualBlocks();

    // GET CURRENT RESIDUALS - WITHOUT APPLICATION OF LOSS FUNCTION
    double cost;
    ceres::Problem::EvaluateOptions options_eval;
    options_eval.apply_loss_function = false;
    std::vector<ceres::ResidualBlockId> single_id;
    std::map<double, Pupil*> residual_per_pupil;
    typedef std::map<double,Pupil*>::const_iterator MapIterator;
    for (auto& pupil: mSupportingPupils){
         if(pupil.ceres_toggle<mStrikes){
                single_id = {pupil.mResidualBlockId};
                options_eval.residual_blocks = single_id;
                problem.Evaluate(options_eval, &cost, NULL, NULL, NULL);
                residual_per_pupil.insert(std::make_pair(cost/sqrt(N_+1), &pupil));
         }
    }

    Eigen::VectorXd mc(2); // SLOPE OF RESIDUALS
    mc[0] = 0;
    mc[1] = 0;

    double alpha;

    // GET AVERAGE COST OF FRACTION OF RESIDUAL BLOCKS
    int Nblocks = static_cast<int>(residual_per_pupil.size()*mResidualsAveragedFraction);

     // SETUP FOR LINEAR REGRESSION ON RESIDUALS AS FUNCTION OF ANGLE
    Eigen::MatrixXd A(Nblocks ,2);
    Eigen::VectorXd b(Nblocks );

    double average_residual_per_block = 0;
    int counter = 0;

    for (MapIterator iter = residual_per_pupil.begin(); iter != residual_per_pupil.end(); iter++){
        if (counter<Nblocks){
            average_residual_per_block += iter->first;
            alpha = acos(-sin(iter->second->optimizedParams[0]) * sin(iter->second->optimizedParams[1]));
            A(counter,0)=alpha;
            A(counter,1)=1.0;
            b(counter)=log10(iter->first);
        }
        ++counter;
    }
    average_residual_per_block /= Nblocks;
    mc = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    if (first_round_number_residual_blocks>mStartRemoveNumber){  // ONLY START REMOVING PUPILS WHEN YOU ALREADY HAVE A FEW

        // REMOVE OUTLIER RESIDUAL BLOCKS
        for (MapIterator iter = residual_per_pupil.begin(); iter != residual_per_pupil.end(); iter++){
          if (iter->second->ceres_toggle<mStrikes){
//                if (iter->first>mOutlierFactor*average_residual_per_block){
                alpha = acos(-sin(iter->second->optimizedParams[0]) * sin(iter->second->optimizedParams[1]));
                if (log10(iter->first)>log10(mOutlierFactor)+mc[0]*alpha+mc[1]){
                    problem.RemoveResidualBlock(iter->second->mResidualBlockId);
                    iter->second->ceres_toggle += 1;
                    if (iter->second->ceres_toggle>=mStrikes){
                        std::cout << "Removing Pupil!" << std::endl;
                    }
                }
            }
        }

    }

    //SAVE ALL RESIDUALBLOCK-IDS IN A VECTOR
    residualblock_vector = std::vector<ceres::ResidualBlockId>();
    problem.GetResidualBlocks(&residualblock_vector);

    // FINAL OPTIMIZATION WITH REMOVED OUTLIERS
    center_weight = mCenterWeightFinal;
    options.max_num_iterations =  mIterationNumbers[4];
    ceres::Solve(options, &problem, &summary);

    // UPDATING MODEL PARAMETERS FROM OPTIMIZATION RESULT
    sphere.center[0] = eye_params[0];
    sphere.center[1] = eye_params[1];
    sphere.center[2] = eye_params[2];
    sphere.radius = sqrt(pow(mEyeballRadius, 2)-pow(mIrisRadius, 2));

    { //SCOPED MUTEX

        std::lock_guard<std::mutex> lock_refraction(mRefractionMutex);

        // CHECKING RESIDUALS
        double cost;
        ceres::Problem::EvaluateOptions options_eval;
        options_eval.apply_loss_function = false;
        std::vector<ceres::ResidualBlockId> single_id;
        mCostPerBlock.clear();

        for (const auto& rb: residualblock_vector){

            single_id = {rb};
            options_eval.residual_blocks = single_id;
            problem.Evaluate(options_eval, &cost, NULL, NULL, NULL);
            mCostPerBlock.push_back(cost/sqrt(N_+1));
            //std::cout << cost/sqrt(N_+1) << std::endl;

        }

        // UPDATE OPTIMIZED PARAMTERS
        mOptimizedParams = std::vector<double>();
        for (int i; i<5; ++i){

            mOptimizedParams.push_back(eye_params[i]);

        }
        for (const auto& pupil: mSupportingPupils){

            if (pupil.ceres_toggle<mStrikes){
                mOptimizedParams.push_back(pupil.optimizedParams[0]);
                mOptimizedParams.push_back(pupil.optimizedParams[1]);
                mOptimizedParams.push_back(pupil.optimizedParams[2]);
            }
        }

        // WRITE RESULTS TO REFRACTION-RESULT OBJECT
        mResult.cost = summary.final_cost;
        mResult.number_of_pupils = first_round_number_residual_blocks; //FINAL RESIDUAL BLOCKS

        mResult.message = summary.message;
        mResult.optimized_center[0] = eye_params[0];
        mResult.optimized_center[1] = eye_params[1];
        mResult.optimized_center[2] = eye_params[2];
        mResult.initial_center[0] = initial[0];
        mResult.initial_center[1] = initial[1];
        mResult.initial_center[2] = initial[2];
        mResult.resFit[0] = mc(0);
        mResult.resFit[1] = mc(1);

        // NEEDED FOR DEBUGGING
        mResult.par_history = callback.get_par_history();
        mResult.cost_history = callback.get_cost_history();
        mResult.pupil_type_history = callback.get_pupil_type_history();
        mResult.circles  = temp_circles;
        mResult.ellipses = temp_ellipses;
        mResult.edge_map = temp_edge_map;

    }

    std::cout<<sphere<<std::endl;

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


// Predictors
Circle EyeModel::predictSingleObservation(std::shared_ptr<Detector2DResult>& observation2D, bool prepare)
{
        if (prepare){

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
            for (cv::Point& p : observation2D->final_edges) {
                p += roi.tl();
                p.x -= image_width_half;
                p.y  = image_height_half - p.y;
            }
        }else{}

        auto newObservationPtr = std::make_shared<const Observation>(observation2D, mFocalLength);
        Circle circle_;
        Circle circle;
        const Circle& unprojectedCircle = selectUnprojectedCircle(mSphere, newObservationPtr->getUnprojectedCirclePair());
        circle = getInitialCircle(mSphere, unprojectedCircle);
        std::pair<PupilParams, double> refraction_result = predictApproximate(mSphere, circle, newObservationPtr);
        circle_ = circleFromParams(mSphere, refraction_result.first);
        return circle_;

}

std::pair<EyeModel::PupilParams, double> EyeModel::predictSwirksi( const Sphere& sphere, const Circle& unrefracted_circle, const ObservationPtr observation ) const
{

        double sx,sy,sz;
        sx = sphere.center[0];
        sy = sphere.center[1];
        sz = sphere.center[2];

        if (sx<0.001){sx+=0.001;}
        if (sy<0.001){sy+=0.001;}

        double xi = -atan2(sy,sx);

        Eigen::Matrix<double, 3, 3> R;
        R << cos(xi),-sin(xi),0,sin(xi),cos(xi),0,0,0,1;
        Eigen::Matrix<double, 3, 3> Rinv;
        Rinv << cos(-xi),-sin(-xi),0,sin(-xi),cos(-xi),0,0,0,1;

        double sx_rot, sy_rot, sz_rot;
        sx_rot = R(0,0)*sx+R(0,1)*sy+R(0,2)*sz;
        sy_rot = R(1,0)*sx+R(1,1)*sy+R(1,2)*sz;
        sz_rot = R(2,0)*sx+R(2,1)*sy+R(2,2)*sz;

        double frame_height = static_cast<double>(observation->getObservation2D()->image_height);
        double frame_width = static_cast<double>(observation->getObservation2D()->image_width);
        double x, y, area;

        x = observation->getObservation2D()->ellipse.center[0]+frame_width/2.0;
        y = frame_height/2.0-observation->getObservation2D()->ellipse.center[1];
        x /= frame_width;
        y /= frame_height;
        y = 1-y;
        area = observation->getObservation2D()->ellipse.area();

        double xscaled = x*frame_width-frame_width/2.0;
        double yscaled = y*frame_height-frame_height/2.0;

        double xrot =  R(0,0)*xscaled+R(0,1)*yscaled;
        double yrot =  R(1,0)*xscaled+R(1,1)*yscaled;

        xrot = (xrot+frame_width/2.)/frame_width;
        yrot = (yrot+frame_height/2.)/frame_height;

        double inputs[5];

        //SCALING OF INPUTS
        inputs[0] = (sx_rot - mConstants[0])/(mConstants[5]);
        inputs[1] = (sz_rot - mConstants[1])/(mConstants[6]);
        inputs[2] = (xrot   - mConstants[2])/(mConstants[7]);
        inputs[3] = (yrot   - mConstants[3])/(mConstants[8]);
        inputs[4] = (area   - mConstants[4])/(mConstants[9]);

        Eigen::Matrix<double, Eigen::Dynamic, 1> polynomial_features;
        polynomial_features.resize(mExponents.rows(),1);
        polynomial_features(0,0) = 1.0;
        for (int i=1;i<mExponents.rows();i++){
            polynomial_features(i,0) = 1.0;
            for (int j=0;j<5;j++){
                if (mExponents(i,j)>0.1){
                   polynomial_features(i,0) *= pow(inputs[j],mExponents(i,j));
                }
           }
        }

        //APPLY LINEAR MAPPING TO POLYNOMIAL FEATURES
        double phi0; // = 0;
        double theta0; // = 0;
        double pupil_radius; // = 0;

        phi0 = mCp.dot(polynomial_features);
        theta0 = mCt.dot(polynomial_features);
        pupil_radius = mCr.dot(polynomial_features);
        phi0         += mConstants[10];
        theta0       += mConstants[11];
        pupil_radius += mConstants[12];

        //ROTATE GAZE VECTOR TO ORIGINAL POSITION OF SPHERE
        Eigen::Matrix<double, 3, 1> gaze_vector_0 = math::sph2cart<double>(1.0, theta0, phi0);
        gaze_vector_0[0] += sx_rot;
        gaze_vector_0[1] += sy_rot;
        gaze_vector_0[2] += sz_rot;

        Eigen::Matrix<double, 3, 1> gaze_vector;
        gaze_vector[0] =  Rinv(0,0)*gaze_vector_0[0]+Rinv(0,1)*gaze_vector_0[1]+Rinv(0,2)*gaze_vector_0[2];
        gaze_vector[1] =  Rinv(1,0)*gaze_vector_0[0]+Rinv(1,1)*gaze_vector_0[1]+Rinv(1,2)*gaze_vector_0[2];
        gaze_vector[2] =  Rinv(2,0)*gaze_vector_0[0]+Rinv(2,1)*gaze_vector_0[1]+Rinv(2,2)*gaze_vector_0[2];

        //TRANSFORM TO SPHERICAL COORDINATES
        Eigen::Matrix<double, 2, 1> gv_spherical = math::cart2sph<double>(gaze_vector[0]-sx, gaze_vector[1]-sy, gaze_vector[2]-sz);
        double phi;
        double theta;
        phi = gv_spherical[1];
        theta = gv_spherical[0];

        //ISTANTIATE PupilParams OBJECT WITH RESULTS
        PupilParams pupilParams = PupilParams(theta, phi, 2*pupil_radius);
        //PupilParams pupilParams = PupilParams(1.5, 1.5, 1.0);

        return  {pupilParams, 0};
}

std::pair<EyeModel::PupilParams, double> EyeModel::predictApproximate( const Sphere& sphere, const Circle& unrefracted_circle, const ObservationPtr observation ) const
{

        double sx,sy,sz;
        sx = sphere.center[0];
        sy = sphere.center[1];
        sz = sphere.center[2];

        if (sx<0.001){sx+=0.001;}
        if (sy<0.001){sy+=0.001;}

        double xi = -atan2(sy,sx);

        Eigen::Matrix<double, 3, 3> R;
        R << cos(xi),-sin(xi),0,sin(xi),cos(xi),0,0,0,1;
        Eigen::Matrix<double, 3, 3> Rinv;
        Rinv << cos(-xi),-sin(-xi),0,sin(-xi),cos(-xi),0,0,0,1;

        double sx_rot, sy_rot, sz_rot;
        sx_rot = R(0,0)*sx+R(0,1)*sy+R(0,2)*sz;
        sy_rot = R(1,0)*sx+R(1,1)*sy+R(1,2)*sz;
        sz_rot = R(2,0)*sx+R(2,1)*sy+R(2,2)*sz;

        double frame_height = static_cast<double>(observation->getObservation2D()->image_height);
        double frame_width = static_cast<double>(observation->getObservation2D()->image_width);
        double x, y, area;

        x = observation->getObservation2D()->ellipse.center[0]+frame_width/2.0;
        y = frame_height/2.0-observation->getObservation2D()->ellipse.center[1];
        x /= frame_width;
        y /= frame_height;
        y = 1-y;
        area = observation->getObservation2D()->ellipse.area();

        double xscaled = x*frame_width-frame_width/2.0;
        double yscaled = y*frame_height-frame_height/2.0;

        double xrot =  R(0,0)*xscaled+R(0,1)*yscaled;
        double yrot =  R(1,0)*xscaled+R(1,1)*yscaled;

        xrot = (xrot+frame_width/2.)/frame_width;
        yrot = (yrot+frame_height/2.)/frame_height;

        double inputs[5];

        //SCALING OF INPUTS
        inputs[0] = (sx_rot - mConstants[0])/(mConstants[5]);
        inputs[1] = (sz_rot - mConstants[1])/(mConstants[6]);
        inputs[2] = (xrot   - mConstants[2])/(mConstants[7]);
        inputs[3] = (yrot   - mConstants[3])/(mConstants[8]);
        inputs[4] = (area   - mConstants[4])/(mConstants[9]);

        Eigen::Matrix<double, Eigen::Dynamic, 1> polynomial_features;
        polynomial_features.resize(mExponents.rows(),1);
        polynomial_features(0,0) = 1.0;
        for (int i=1;i<mExponents.rows();i++){
            polynomial_features(i,0) = 1.0;
            for (int j=0;j<5;j++){
                if (mExponents(i,j)>0.1){
                   polynomial_features(i,0) *= pow(inputs[j],mExponents(i,j));
                }
           }
        }

        //APPLY LINEAR MAPPING TO POLYNOMIAL FEATURES
        double phi0; // = 0;
        double theta0; // = 0;
        double pupil_radius; // = 0;

        phi0 = mCp.dot(polynomial_features);
        theta0 = mCt.dot(polynomial_features);
        pupil_radius = mCr.dot(polynomial_features);
        phi0         += mConstants[10];
        theta0       += mConstants[11];
        pupil_radius += mConstants[12];

        //ROTATE GAZE VECTOR TO ORIGINAL POSITION OF SPHERE
        Eigen::Matrix<double, 3, 1> gaze_vector_0 = math::sph2cart<double>(1.0, theta0, phi0);
        gaze_vector_0[0] += sx_rot;
        gaze_vector_0[1] += sy_rot;
        gaze_vector_0[2] += sz_rot;

        Eigen::Matrix<double, 3, 1> gaze_vector;
        gaze_vector[0] =  Rinv(0,0)*gaze_vector_0[0]+Rinv(0,1)*gaze_vector_0[1]+Rinv(0,2)*gaze_vector_0[2];
        gaze_vector[1] =  Rinv(1,0)*gaze_vector_0[0]+Rinv(1,1)*gaze_vector_0[1]+Rinv(1,2)*gaze_vector_0[2];
        gaze_vector[2] =  Rinv(2,0)*gaze_vector_0[0]+Rinv(2,1)*gaze_vector_0[1]+Rinv(2,2)*gaze_vector_0[2];

        //TRANSFORM TO SPHERICAL COORDINATES
        Eigen::Matrix<double, 2, 1> gv_spherical = math::cart2sph<double>(gaze_vector[0]-sx, gaze_vector[1]-sy, gaze_vector[2]-sz);
        double phi;
        double theta;
        phi = gv_spherical[1];
        theta = gv_spherical[0];

        //ISTANTIATE PupilParams OBJECT WITH RESULTS
        PupilParams pupilParams = PupilParams(theta, phi, 2*pupil_radius);
        //PupilParams pupilParams = PupilParams(1.5, 1.5, 1.0);

        return  {pupilParams, 0};
}

std::pair<EyeModel::PupilParams, double> EyeModel::predictFullOptimization( const Sphere& sphere, const Circle& unrefracted_circle, const ObservationPtr observation ) const
{

      // PARAMTERS FOR COSTFUNCTION
      double center_weight;
      double * const center_weight_ptr = &center_weight;

      Eigen::Matrix<double, Eigen::Dynamic, 1> par;
      par = Eigen::Matrix<double, Eigen::Dynamic, 1>(8);
      Eigen::Matrix<double, 3, 1> pupil_radial = unrefracted_circle.center-sphere.center;
      Eigen::Matrix<double, 2, 1> pupil_spherical = math::cart2sph(pupil_radial[0], pupil_radial[1], pupil_radial[2]);

      par.segment<3>(0) = mSphere.center;
      par[3] = mCorneaRadius;
      par[4] = mIrisRadius;

      if (true){  // CORRECTION WITH PHENOMENOLOGICAL CONSTANTS

          Eigen::Matrix<double, 3, 1> v2 = math::sph2cart<double>(1.0, pupil_spherical[0], pupil_spherical[1]);
          Eigen::Matrix<double, 3, 1> v1 = -sphere.center;
          v1.normalize();
          double alpha = 0.17 * acos(v1.dot(v2));
          Eigen::Matrix<double, 3, 3> R = correction_matrix(v1, v2, alpha);
          v1 = R*v2;
          Eigen::Matrix<double, 2, 1> corrected_spherical = math::cart2sph(v1[0],v1[1],v1[2]);
          double corrected_radius = 1./sqrt(0.55*pow(alpha, 2)-0.08*alpha+1.4)*unrefracted_circle.radius;

          par[5] = corrected_spherical[0];
          par[6] = corrected_spherical[1];
          par[7] = corrected_radius;

      }else{

          par[5] = pupil_spherical[0];
          par[6] = pupil_spherical[1];
          par[7] = unrefracted_circle.radius;

      }

      if ( par(7,0) < 0.7 ) par(7,0) = 0.70001;
      if ( par(7,0) > 5.0 ) par(7,0) = 4.99999;

      const auto& pupilInliers = observation->getObservation2D()->final_edges;
      const Ellipse& ellipse =  observation->getObservation2D()->ellipse;
      cv::Point ellipse_center(ellipse.center[0], ellipse.center[1]);

      ceres::Problem problem;

      int N_;
      if (mEdgeNumber==-1){
          N_ = pupilInliers.size();
      }else{
          N_ = 2*mEdgeNumber < pupilInliers.size() ? 2*mEdgeNumber : pupilInliers.size();
      }

      ceres::ResidualBlockId res_id;
      res_id = problem.AddResidualBlock(new ceres::AutoDiffCostFunction < RefractionResidualFunction<double>, ceres::DYNAMIC, 1, 1, 1, 2, 3 > (
                               new RefractionResidualFunction<double>(pupilInliers, mEyeballRadius, mFocalLength, ellipse_center, center_weight_ptr, N_),
                               N_+1),
                               new ceres::CauchyLoss(mCauchyLossScale), &par[0], &par[1], &par[2], &par[3], &par[5]);

      problem.SetParameterBlockConstant(&par[0]);
      problem.SetParameterBlockConstant(&par[1]);
      problem.SetParameterBlockConstant(&par[2]);
      problem.SetParameterBlockConstant(&par[3]);
      problem.SetParameterLowerBound(&par[5], 2, 0.7);
      problem.SetParameterUpperBound(&par[5], 2, 5.0);

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = false;
      options.function_tolerance = 1e-22;
      options.num_threads = 2;
      ceres::Solver::Summary summary;


      center_weight = mCenterWeightInitial;
      options.max_num_iterations = 40;
      ceres::Solve(options, &problem, &summary);

      center_weight = mCenterWeightFinal;
      options.max_num_iterations = 20;
      ceres::Solve(options, &problem, &summary);

      PupilParams pupilParams = PupilParams(par(5,0), par(6,0), par(7,0)/2.0);

      double cost;
      ceres::Problem::EvaluateOptions options_eval;
      options_eval.apply_loss_function = false;
      std::vector<ceres::ResidualBlockId> single_id;
      single_id = {res_id};
      options_eval.residual_blocks = single_id;
      problem.Evaluate(options_eval, &cost, NULL, NULL, NULL);
      return  {pupilParams,cost/sqrt(2*N_+1)};

}


// Setter
void EyeModel::setSphereCenter(std::vector<double> center)
{

    mSphere.center[0] = center[0];
    mSphere.center[1] = center[1];
    mSphere.center[2] = center[2];
    eye_params[0] = center[0];
    eye_params[1] = center[1];
    eye_params[2] = center[2];
    mSphere.radius = sqrt(pow(mEyeballRadius,2)-pow(mIrisRadius,2));

}


// Getter
std::vector<double> EyeModel::getCostPerPupil() const
{
        std::lock_guard<std::mutex> lock_refraction(mRefractionMutex);
        return mCostPerBlock;

}

std::vector<double> EyeModel::getOptimizedParameters() const
{
        std::lock_guard<std::mutex> lock_refraction(mRefractionMutex);
        return mOptimizedParams;

}

EyeModel::Sphere EyeModel::getSphere() const
{
    std::lock_guard<std::mutex> lockModel(mModelMutex);
    return mSphere;
}

EyeModel::Sphere EyeModel::getInitialSphere() const
{
    std::lock_guard<std::mutex> lockModel(mModelMutex);
    return mInitialSphere;
}

Detector3DResultRefraction EyeModel::getRefractionResult() const
{
    std::lock_guard<std::mutex> lock_refraction(mRefractionMutex);
    return mResult;
}

std::vector<double> EyeModel::getResFit() const
{
    std::vector<double> temp = {mResult.resFit[0], mResult.resFit[1]};
    return temp;
}


// Utilities
void EyeModel::setApproximationParameters(std::vector<double> Cp, std::vector<double> Ct, std::vector<double> Cr, std::vector<double> exponents, std::vector<double> constants)
{

        mCp.resize(Cp.size(),1);
        mCt.resize(Ct.size(),1);
        mCr.resize(Cr.size(),1);

        for (int i; i<Cp.size();i++){

            mCp(i,0) = Cp[i];
            mCt(i,0) = Ct[i];
            mCr(i,0) = Cr[i];

        }

        mExponents.resize(exponents.size()/5,5);
        for (int i; i<exponents.size();i++){
            mExponents(i/5,i%5) = exponents[i];
        }

        mConstants.resize(constants.size(),1);
        for (int i; i<constants.size();i++){
            mConstants(i,0) = constants[i];
        }

}

int EyeModel::addObservation(std::shared_ptr<Detector2DResult>& observation2D, int prepare_toggle)
{
    if (prepare_toggle){

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
            for (cv::Point& p : observation2D->final_edges) {
                p += roi.tl();
                p.x -= image_width_half;
                p.y  = image_height_half - p.y;
            }

        }

    auto newObservationPtr = std::make_shared<const Observation>(observation2D, mFocalLength);
    if (newObservationPtr->getObservation2D()->confidence>0.4){
        mSupportingPupilsToAdd.emplace_back(newObservationPtr);
        for( auto& pupil : mSupportingPupilsToAdd){
            mSupportingPupils.push_back(std::move(pupil));
        }
        mSupportingPupilsToAdd.clear();
        mSupportingPupilSize = mSupportingPupils.size();
    }else{
        std::cout << "Low confidence: " << newObservationPtr->getObservation2D()->confidence << std::endl;
    }

    return static_cast<int>(mSupportingPupils.size());
}

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

Eigen::Matrix<double,3,3> EyeModel::correction_matrix(Eigen::Matrix<double,3,1> v1, Eigen::Matrix<double,3,1> v2, double theta) const
{

    Eigen::Matrix<double,3,3> R;
    Eigen::Matrix<double,3,1> axis;
    v1.normalize();
    v2.normalize();
    axis = v1.cross(v2);
    axis.normalize();
    double a = cos(theta/2.0);
    double b,c,d,aa,bb,cc,dd,bc,ad,ac,ab,bd,cd;
    b = -axis[0]*sin(theta/2.0);
    c = -axis[1]*sin(theta/2.0);
    d = -axis[2]*sin(theta/2.0);
    aa = pow(a,2);
    bb = pow(b,2);
    cc = pow(c,2);
    dd = pow(d,2);
    bc = b*c;
    ad = a*d;
    ac = a*c;
    ab = a*b;
    bd = b*d;
    cd = c*d;

    R << aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac),
         2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab),
         2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc;

    return R;

}

void EyeModel::reset()
{};

}

// singleeyefitter
