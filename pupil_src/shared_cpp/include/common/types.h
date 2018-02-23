
#ifndef singleeyefitter_types_h__
#define singleeyefitter_types_h__

#include "geometry/Ellipse.h"
#include "geometry/Circle.h"
#include "geometry/Sphere.h"
#include "projection.h"

#include <vector>
#include <memory>
#include <chrono>
#include <map>

#include <opencv2/core.hpp>


namespace singleeyefitter {


    //########  2D Detector ############
    typedef std::vector<std::vector<cv::Point> > Contours_2D;
    typedef std::vector<cv::Point> Contour_2D;
    typedef std::vector<cv::Point> Edges2D;
    typedef std::vector<int> ContourIndices;
    typedef Ellipse2D<double> Ellipse;

    //########  3D Detector ############

    typedef Eigen::Matrix<double, 2, 1> Vector2;
    typedef Eigen::Matrix<double, 3, 1> Vector3;
    typedef Eigen::ParametrizedLine<double, 2> Line;
    typedef Eigen::ParametrizedLine<double, 3> Line3;
    typedef Circle3D<double> Circle;
    typedef size_t Index;

    typedef std::vector<Vector3> Contour3D;
    typedef std::vector<Vector3> Edges3D;
    typedef std::vector<std::vector<Vector3>> Contours3D;

    struct ConfidenceValue{
        ConfidenceValue(double v,double c)
        {
            value = v;
            confidence = c;
        };
        ConfidenceValue()
        {
            value = 0;
            confidence = 0;
        };
        double value;
        double confidence;
        };

    // general time
    typedef std::chrono::steady_clock Clock;


    // every coordinates are relative to the roi
    struct Detector2DResult {
        double confidence =  0.0 ;
        Ellipse ellipse = Ellipse::Null;
        Edges2D final_edges; // edges used to fit the final ellipse in 2D
        Edges2D raw_edges;
        cv::Rect current_roi; // contains the roi for this results
        double timestamp = 0.0;
        int image_width = 0;
        int image_height = 0;

    };

    struct Detector3DResultRefraction {

        double initial_center[3];
        double optimized_center[3];
        double cost;
        int number_of_pupils;
        std::vector<std::vector<double>> par_history;
        std::vector<std::vector<int>> pupil_type_history;
        std::vector<double> cost_history;
//        std::vector<double> residual_histogram;
//        double mean_residual;
//        double std_residual;
        std::string message;
//        std::map<int,std::vector<std::vector<double>>> edge_map;
//        std::vector<Circle> circles;
//        std::vector<Ellipse> ellipses;
        double resFit[2];

    };

    struct ModelDebugProperties{

        std::vector<double> optimizedParameters;
        std::vector<double> costPerPupil;
        std::vector<double> resFit;
        Sphere<double> sphere;
        Sphere<double> initialSphere;
        std::vector<Vector3> binPositions;
        double maturity;
        double solverFit;
        double confidence;
        double performance;
        double performanceGradient;
        int modelID;
        double birthTimestamp;
    };

    struct Detector3DResult {
        double confidence =  0.0 ;
        Circle circle  = Circle::Null;
        double cost;
        Ellipse ellipse = Ellipse::Null; // the circle projected back to 2D
        Sphere<double> sphere = Sphere<double>::Null;
        Ellipse projectedSphere = Ellipse::Null; // the sphere projected back to 2D
        double timestamp;
        int modelID = 0;
        double modelBirthTimestamp = 0.0;
        double modelConfidence = 0.0;
        //-------- For visualization ----------------
        // just valid if we want it for visualization
        Edges3D edges;
        Circle predictedCircle = Circle::Null;
        std::vector<ModelDebugProperties> models;
        Detector3DResultRefraction RefractionResult;

    };

    // use a struct for all properties and pass it to detect method every time we call it.
    // Thus we don't need to keep track if GUI is updated and cython handles conversion from Dict to struct
    struct Detector2DProperties {
        int intensity_range;
        int blur_size;
        float canny_treshold;
        float canny_ration;
        int canny_aperture;
        int pupil_size_max;
        int pupil_size_min;
        float strong_perimeter_ratio_range_min;
        float strong_perimeter_ratio_range_max;
        float strong_area_ratio_range_min;
        float strong_area_ratio_range_max;
        int contour_size_min;
        float ellipse_roundness_ratio;
        float initial_ellipse_fit_treshhold;
        float final_perimeter_ratio_range_min;
        float final_perimeter_ratio_range_max;
        float ellipse_true_support_min_dist;
        float support_ratio_weight;
        bool filter_solutions;
        bool take_maximum;
        bool support_from_raw_edges;

    };

    enum refraction_mode {SWIRSKI, REFRACTION, REFRACTION_APPROXIMATE};

    struct Detector3DProperties {
         int edge_number;
         int strikes;
         double center_weight_initial;
         double center_weight_final;
         std::vector<int> iteration_numbers;
         double residuals_averaged_fraction;
         double outlier_factor;
         int start_remove_number;
         double cauchy_loss_scale;
         double eyeball_radius;
         double cornea_radius;
         double iris_radius;
         double n_ref;
         refraction_mode run_mode;
         std::vector<int> pars_to_optimize;
    };


} // singleeyefitter namespace

#endif //singleeyefitter_types_h__
