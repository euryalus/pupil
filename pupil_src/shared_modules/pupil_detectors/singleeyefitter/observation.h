#ifndef OBSERVATION_H__
#define OBSERVATION_H__

namespace singleeyefitter {

    class Observation {
        std::shared_ptr<const Detector2DResult> mObservation2D;
        std::pair<Circle,Circle> mUnprojectedCirclePair;
        Line mProjectedCircleGaze;

    public:

        Observation(std::shared_ptr<const Detector2DResult> observation, double focalLength) : mObservation2D(observation) {
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

        Observation(const Observation& that) = delete; // forbid copying
        Observation(Observation&& that) = delete; // forbid moving
        Observation() = delete; // forbid default construction

        const std::shared_ptr<const Detector2DResult> getObservation2D() const { return mObservation2D;};
        const std::pair<Circle,Circle>& getUnprojectedCirclePair() const { return mUnprojectedCirclePair; };
        const Line& getProjectedCircleGaze() const { return mProjectedCircleGaze; };

    };

}

#endif /* end of include guard: OBSERVATION_H__ */
