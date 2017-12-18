#ifndef EYEMODEL_H__
#define EYEMODEL_H__

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

#include "common/types.h"
#include "mathHelper.h"
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <list>
#include <atomic>

namespace singleeyefitter {

    class Observation {
        /*
            Observation class

            Hold data which is precalculated for every new observation
            Every observation is shared between different models

        */
        std::shared_ptr<const Detector2DResult> mObservation2D;
        std::pair<Circle,Circle> mUnprojectedCirclePair;
        Line mProjectedCircleGaze;


    public:
        Observation(std::shared_ptr<const Detector2DResult> observation, double focalLength) :
            mObservation2D(observation)
        {
                const double circleRadius = 1.0;
                // Do a per-image unprojection of the pupil ellipse into the two fixed
                // sized circles that would project onto it. The size of the circles
                // doesn't matter here, only their center and normal does.
                mUnprojectedCirclePair = unproject(mObservation2D->ellipse, circleRadius , focalLength);
                 // Get projected circles and gaze vectors
                //
                // Project the circle centers and gaze vectors down back onto the image
                // plane. We're only using them as line parametrisations, so it doesn't
                // matter which of the two centers/gaze vectors we use, as the
                // two gazes are parallel and the centers are co-linear.
                const auto& c = mUnprojectedCirclePair.first.center;
                const auto& v = mUnprojectedCirclePair.first.normal;
                Vector2 cProj = project(c, focalLength);
                Vector2 vProj = project(v + c, focalLength) - cProj;
                vProj.normalize();
                mProjectedCircleGaze = Line(cProj, vProj);

        }
        Observation( const Observation& that ) = delete; // forbid copying
        Observation( Observation&& that ) = delete; // forbid moving
        Observation() = delete; // forbid default construction
        const std::shared_ptr<const Detector2DResult> getObservation2D() const { return mObservation2D;};
        const std::pair<Circle,Circle>& getUnprojectedCirclePair() const { return mUnprojectedCirclePair; };
        const Line& getProjectedCircleGaze() const { return mProjectedCircleGaze; };

    };

    typedef std::shared_ptr<const Observation> ObservationPtr;

    //enum refraction_mode {SWIRSKI, REFRACTION, REFRACTION_APPROXIMATE};

    class EyeModel {

            typedef singleeyefitter::Sphere<double> Sphere;

            public:

                // Constructors
                EyeModel(int modelId, double timestamp, double focalLength, Vector3 cameraCenter = Vector3::Zero(), int initialUncheckedPupils = 5, double binResolution = 0.05);
                EyeModel(const EyeModel&) = delete;
                ~EyeModel();

                // General functions
                Detector3DResult predictAndUpdate(std::shared_ptr<Detector2DResult>&, const Detector3DProperties&, bool);
                std::pair<Circle, double> presentObservation(const ObservationPtr, const Detector3DProperties&);
                int addObservation(std::shared_ptr<Detector2DResult>&, int);
                Detector3DResultRefraction optimize(bool, const Detector3DProperties&);
                Circle predictSingleObservation(std::shared_ptr<Detector2DResult>&, bool, const Detector3DProperties&);
                void setApproximationParameters(std::vector<double>, std::vector<double>, std::vector<double> , std::vector<double> , std::vector<double>);
                void reset();

                // Setter
                void setSphereCenter(std::vector<double>,const Detector3DProperties&);

                // Getter
                Sphere getSphere() const;
                Sphere getInitialSphere() const;
                std::vector<double> getOptimizedParameters() const;
                std::vector<double> getCostPerPupil() const;
                std::vector<double> getResFit() const;
                int getNumResidualBlocks() const;
                Detector3DResultRefraction getRefractionResult() const;
                double getFocalLength(){return mFocalLength;};
                int getModelID() const {return mModelID;};
                double getBirthTimestamp() const {return mBirthTimestamp;};

            private:

                struct PupilParams {

                    double theta, psi, radius;
                    PupilParams() : theta(0), psi(0), radius(0) {};
                    PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius){};

                };

                struct Pupil {

                    Circle mCircle;
                    PupilParams mParams;
                    const ObservationPtr mObservationPtr;
                    //Pupil( const ObservationPtr observationPtr ) : mObservationPtr( observationPtr ){};
                    Pupil( const ObservationPtr observationPtr) : mObservationPtr( observationPtr ){
                        mParams = PupilParams(0,0,0);
                        optimizedParams[0] = 0;
                        optimizedParams[1] = 0;
                        optimizedParams[2] = 0;
                    };
                    Pupil( const ObservationPtr observationPtr, PupilParams params ) : mObservationPtr( observationPtr ){
                        mParams = PupilParams(params.theta, params.psi, params.radius);
                        optimizedParams[0] = params.theta;
                        optimizedParams[1] = params.psi;
                        optimizedParams[2] = params.radius;
                    };
                    ceres::ResidualBlockId mResidualBlockId;
                    double * const optimizedParams = static_cast<double * const>(malloc(3*sizeof(double)));
                    int ceres_toggle = 0;

                };

                Sphere findSphereCenter(const Detector3DProperties&, bool use_ransac = true);
                Sphere initialiseModel(const Detector3DProperties& props);
                double refineWithEdgesRefraction(Sphere&, const Detector3DProperties&);
                double refineWithEdgesSwirski(Sphere&, const Detector3DProperties&);
                bool tryTransferNewObservations();

                Detector3DResultRefraction mResult;

                // General functions
                bool isSpatialRelevant(const Circle& circle);
                void initialiseSingleObservation(const Sphere& sphere, Pupil& pupil) const;

                // Predictors
                std::pair<PupilParams, double> predictRefraction( const Sphere& sphere, const Circle& unrefracted_circle, const ObservationPtr observation, const Detector3DProperties&) const;
                std::pair<PupilParams, double> predictRefractionApproximate(const Sphere& sphere, const Circle& unrefracted_circle, const ObservationPtr observation, const Detector3DProperties&) const;
                std::pair<PupilParams, double> predictSwirski(const Sphere& sphere, const Circle& circle, const ObservationPtr observation, const Detector3DProperties&) const;

                // Utilities
                const Circle& selectUnprojectedCircle(const Sphere& sphere, const std::pair<const Circle, const Circle>& circles) const;
                Circle circleFromParams(const Sphere& eye, const  PupilParams& params) const;
                void prepareObservation(std::shared_ptr<Detector2DResult>&) const;
                Circle getIntersectedCircle(const Sphere& sphere, const Circle& circle) const;
                Circle getInitialCircle(const Sphere& sphere, const Circle& circle) const;
                Eigen::Matrix<double,3,3> correction_matrix(Eigen::Matrix<double,3,1> v1, Eigen::Matrix<double,3,1> v2, double theta) const;

                // Spatial bins
                std::unordered_map<Vector2, bool, math::matrix_hash<Vector2>> mSpatialBins;
                std::vector<Vector3> mBinPositions; // for visualization

                // Concurrency
                mutable std::mutex mModelMutex;
                std::mutex mPupilMutex;
                mutable std::mutex mRefractionMutex;
                std::thread mWorker;
                Clock::time_point mLastModelRefinementTime;

                // General model parameters
                const double mFocalLength;
                const Vector3 mCameraCenter;
                const int mInitialUncheckedPupils;
                const double mBinResolution;
                const int mTotalBins;
                const int mModelID;
                double mBirthTimestamp;

                // Flag for debug mode
                bool mDebug;

                // More general model and optimization parameters
//                refraction_mode mRefractionMode;
//                double mEdgeNumber;
//                double mStrikes;
//                double mCenterWeightInitial;
//                double mCenterWeightFinal;
//                int mIterationNumbers[5];
//                double mResidualsAveragedFraction;
//                double mOutlierFactor;
//                int mStartRemoveNumber;
//                double mCauchyLossScale;
//                double mEyeballRadius;
//                double mCorneaRadius;
//                double mIrisRadius;

                // Parameters of approximation function
                Eigen::Matrix<double, Eigen::Dynamic, 1> mCp;
                Eigen::Matrix<double, Eigen::Dynamic, 1> mCt;
                Eigen::Matrix<double, Eigen::Dynamic, 1> mCr;
                Eigen::Matrix<int, Eigen::Dynamic, 5> mExponents;
                Eigen::Matrix<double, Eigen::Dynamic, 1> mConstants;

                // Ceres related
                std::vector<double> mCostPerBlock;
                double * const eye_params = static_cast<double * const>(malloc(5*sizeof(double)));
                std::vector<double> mOptimizedParams;
                void removePupilFromOptimization(std::vector<Pupil>::iterator iter);

                // Thread sensitive variables
                double mSolverFit;                    // Residual of Ceres solver, thread sensitive
                Sphere mSphere;                       // Thread sensitive
                Sphere mInitialSphere;                // Thread sensitive
                std::vector<Pupil> mSupportingPupils; // just used within the worker thread, Thread sensitive
                int mSupportingPupilSize;             // Thread sensitive, use this to get the SupportedPupil size

                // Observations are saved here and only if needed transferred to mSupportingPupils since mSupportingPupils needs a mutex
                std::vector<Pupil> mSupportingPupilsToAdd;

    };

} // singleeyefitter

#endif /* end of include guard: EYEMODEL_H__ */
