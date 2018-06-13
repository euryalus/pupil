#ifndef EYEMODEL_V2_H__
#define EYEMODEL_V2_H__

#include "common/types.h"
#include "mathHelper.h"
#include "observation.h" //Needed at this stage for definition of Observation class
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <list>
#include <atomic>

#include <Eigen/Dense>

namespace singleeyefitter {

    typedef std::shared_ptr<const Observation> ObservationPtr;

    class EyeModel_v2 {

            typedef singleeyefitter::Sphere<double> Sphere;

        public:

             EyeModel_v2(double timestamp, double focalLength, Vector3 cameraCenter);
             EyeModel_v2(const EyeModel_v2&) = delete;
            ~EyeModel_v2();

            int addObservation(std::shared_ptr<Detector2DResult>& observation2D, bool prepare_toggle=true);
            Sphere optimizeModel();
            void reset();

            // Getter
            Sphere getSphere() const {return mSphere;};
            Sphere getInitialSphere() const {return mInitialSphere;};
            double getBirthTimestamp() const {return mBirthTimestamp;};
            int getSupportingPupilsSize() const {return mSupportingPupilsSize;};
            double getFocalLength() const {return mFocalLength;};

        private:

            struct PupilParams {
                double theta, psi, radius;
                PupilParams() : theta(0), psi(0), radius(0){};
                PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius){};
            };

            struct Pupil {
                Circle mCircle;
                PupilParams mParams;
                const ObservationPtr mObservationPtr;
                Pupil(const ObservationPtr observationPtr) : mObservationPtr(observationPtr){};
            };

            Sphere findSphereCenter(bool use_ransac=true);
            Sphere initialiseModel();

            const Circle& selectUnprojectedCircle(const Sphere& sphere, const std::pair<const Circle, const Circle>& circles) const;
            Circle getIntersectedCircle(const Sphere& sphere, const Circle& circle) const;
            Circle circleFromParams(const Sphere& eye, const PupilParams& params) const;

            const double mFocalLength;
            const Vector3 mCameraCenter;
            double mBirthTimestamp;

            Sphere mSphere;
            Sphere mInitialSphere;
            std::vector<Pupil> mSupportingPupils;
            int mSupportingPupilsSize;

            Eigen::Matrix<double,2,Eigen::Dynamic> pupil_centers;
            Eigen::Matrix<double,3,Eigen::Dynamic> gaze_vector_candidates_1;
            Eigen::Matrix<double,3,Eigen::Dynamic> gaze_vector_candidates_2;

             // Utilities
            void prepareObservation(std::shared_ptr<Detector2DResult>& observation2D) const;

    };


} // singleeyefitter

#endif /* end of include guard: EYEMODEL_V2_H__ */
