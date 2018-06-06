#ifndef EYEMODEL_H__
#define EYEMODEL_H__

#include "common/types.h"
#include "observation.h" //Needed at this stage for definition of Observation class
#include "mathHelper.h"
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <list>
#include <atomic>

namespace singleeyefitter {

    typedef std::shared_ptr<const Observation> ObservationPtr;

    class EyeModel {

            typedef singleeyefitter::Sphere<double> Sphere;
        public:

            EyeModel( int modelId, double timestamp,  double focalLength, Vector3 cameraCenter, int initialUncheckedPupils = 3, double binResolution = 0.05  );
            EyeModel(const EyeModel&) = delete;
            //EyeModel(EyeModel&&); // we need a explicit 1/Move constructor because of the mutex
            ~EyeModel();


            std::pair<Circle,ConfidenceValue> presentObservation(const ObservationPtr observation, double averageFramerate  );
            Sphere getSphere() const;
            Sphere getInitialSphere() const;

            // how sensitive the model is for wrong observations
            // if we have to many wring observation new models are created
            // void setSensitivity( float sensitivity ); NOT USED

            // Describing how good different properties of the Eye are
            double getMaturity() const ; // How much spatial variance there is
            double getConfidence() const; // How much do we believe in this model
            double getPerformance() const; // How much do we believe in this model
            double getPerformanceGradient() const;
            double getSolverFit() const ; // The residual of the sphere calculation

            int getModelID() const { return mModelID; };
            double getBirthTimestamp() const { return mBirthTimestamp; };

            // ----- Visualization --------
            std::vector<Vector3> getBinPositions() const {return mBinPositions;};
            // ----- Visualization END --------


        private:


            struct PupilParams {
                double theta, psi, radius;
                PupilParams() : theta(0), psi(0), radius(0) {};
                PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius){};
            };

            struct Pupil{
                Circle mCircle;
                PupilParams mParams;
                const ObservationPtr mObservationPtr;
                Pupil( const ObservationPtr observationPtr ) : mObservationPtr( observationPtr ){};
            };

            Sphere findSphereCenter( bool use_ransac = true);
            Sphere initialiseModel();
            double refineWithEdges( Sphere& sphere  );
            bool tryTransferNewObservations();

            ConfidenceValue calculateModelOberservationFit(const Circle&  unprojectedCircle, const Circle& initialisedCircle, double confidence) const;
            void updatePerformance( const ConfidenceValue& observation_fit,  double averageFramerate);

            double calculateModelFit(const Circle&  unprojectedCircle, const Circle& optimizedCircle) const;
            bool isSpatialRelevant(const Circle& circle);

            const Circle& selectUnprojectedCircle(const Sphere& sphere, const std::pair<const Circle, const Circle>& circles) const;
            void initialiseSingleObservation( const Sphere& sphere, Pupil& pupil) const;
            Circle getIntersectedCircle( const Sphere& sphere, const Circle& circle) const;

            //Circle circleFromParams( CircleParams& params) const;
            Circle circleFromParams(const Sphere& eye,const  PupilParams& params) const;



            std::unordered_map<Vector2, bool, math::matrix_hash<Vector2>> mSpatialBins;
            std::vector<Vector3> mBinPositions; // for visualization

            mutable std::mutex mModelMutex;
            std::mutex mPupilMutex;
            std::thread mWorker;
            Clock::time_point mLastModelRefinementTime;


            // Factors which describe how good certain properties of the model are
            //std::list<double> mModelSupports; // values to calculate the average
            double mSolverFit; // Residual of Ceres sovler , Thread sensitive
            math::WMA<double> mPerformance; // moving Average of model support
            float mPerformanceWindowSize;  // in seconds
            double mPerformanceGradient;
            Clock::time_point mLastPerformanceCalculationTime; // for the gradient calculation when need keep track of the time

            const double mFocalLength;
            const Vector3 mCameraCenter;
            const int mInitialUncheckedPupils;
            const double mBinResolution;
            const int mTotalBins;
            const int mModelID;
            double mBirthTimestamp;

            Sphere mSphere;   // Thread sensitive
            Sphere mInitialSphere;    // Thread sensitive
            std::vector<Pupil> mSupportingPupils; // just used within the worker thread, Thread sensitive
            int mSupportingPupilSize; // Thread sensitive, use this to get the SupportedPupil size

            // observations are saved here and only if needed transfered to mObservation
            // since mObservations needs a mutex
            std::vector<Pupil> mSupportingPupilsToAdd;
    };

} // singleeyefitter

#endif /* end of include guard: EYEMODEL_H__ */
