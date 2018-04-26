'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int32_t
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool


cdef extern from '<opencv2/core.hpp>':

  int CV_8UC1
  int CV_8UC3

cdef extern from '<opencv2/core.hpp>' namespace 'cv':

  cdef cppclass Mat :
      Mat() except +
      Mat( int height, int width, int type, void* data  ) except+
      Mat( int height, int width, int type ) except+

cdef extern from '<opencv2/core.hpp>' namespace 'cv':

  cdef cppclass Rect_[T]:
    Rect_() except +
    Rect_( T x, T y, T width, T height ) except +
    T x, y, width, height

cdef extern from '<opencv2/core.hpp>' namespace 'cv':

  cdef cppclass Point_[T]:
    Point_() except +
    T x
    T y

cdef extern from '<opencv2/core.hpp>' namespace 'cv':

  cdef cppclass Scalar_[T]:
    Scalar_() except +
    Scalar_( T x ) except +

cdef extern from '<Eigen/Eigen>' namespace 'Eigen':

    cdef cppclass Matrix21d "Eigen::Matrix<double,2,1>": # eigen defaults to column major layout
        Matrix21d() except +
        double * data()
        double& operator[](size_t)

    cdef cppclass Matrix31d "Eigen::Matrix<double,3,1>": # eigen defaults to column major layout
        Matrix31d() except +
        Matrix31d(double x, double y, double z)
        double * data()
        double& operator[](size_t)
        bint isZero()


cdef extern from 'common/types.h':

    cdef cppclass Ellipse2D[T]:
        Ellipse2D()
        Ellipse2D(T x, T y, T major_radius, T minor_radius, T angle) except +
        Matrix21d center
        T major_radius
        T minor_radius
        T angle

    cdef cppclass Sphere[T]:
        Matrix31d center
        T radius

    cdef cppclass Circle3D[T]:
        Matrix31d center
        Matrix31d normal
        float radius

    #typdefs
    ctypedef Matrix31d Vector3
    ctypedef Matrix21d Vector2
    ctypedef vector[vector[Vector3]] Contours3D
    ctypedef vector[Vector3] Edges3D
    ctypedef vector[Point_[int]] Edges2D
    ctypedef vector[vector[Point_[int]]] Contours_2D
    ctypedef vector[Point_[int]] Contour_2D
    ctypedef Circle3D[double] Circle
    ctypedef Ellipse2D[double] Ellipse

    cdef struct Detector2DResult:
        double confidence
        Ellipse ellipse
        Edges2D final_edges
        Edges2D raw_edges
        Rect_[int] current_roi
        double timestamp
        int image_width
        int image_height

    cdef struct Detector3DResultRefraction:
        double initial_center[3]
        double optimized_center[3]
        double cost
        int number_of_pupils
        vector[vector[double]] par_history
        vector[vector[int]] pupil_type_history
        vector[double] cost_history
        #vector[double] residual_histogram
        #double mean_residual
        #double std_residual
        #map[int,vector[vector[double]]] edge_map
        #vector[Circle] circles
        #vector[Ellipse] ellipses
        double resFit[2]


    cdef struct ModelDebugProperties:
        vector[double] optimizedParameters
        vector[double] costPerPupil
        vector[double] resFit
        Sphere[double] sphere
        Sphere[double] initialSphere
        vector[Vector3] binPositions
        double maturity
        double solverFit
        double confidence
        double performance
        double performanceGradient
        int modelID
        double birthTimestamp


    cdef struct Detector3DResult:
        double timestamp
        Circle circle
        double cost
        Ellipse ellipse
        Sphere[double] sphere
        Ellipse projectedSphere
        double confidence
        double modelConfidence
        int modelID
        double modelBirthTimestamp
        #-------- For visualization ----------------
        Detector3DResultRefraction RefractionResult
        Edges3D edges
        Circle predictedCircle
        vector[ModelDebugProperties] models

    cdef struct Detector2DProperties:
        int intensity_range
        int blur_size
        float canny_treshold
        float canny_ration
        int canny_aperture
        int pupil_size_max
        int pupil_size_min
        float strong_perimeter_ratio_range_min
        float strong_perimeter_ratio_range_max
        float strong_area_ratio_range_min
        float strong_area_ratio_range_max
        int contour_size_min
        float ellipse_roundness_ratio
        float initial_ellipse_fit_treshhold
        float final_perimeter_ratio_range_min
        float final_perimeter_ratio_range_max
        float ellipse_true_support_min_dist
        float support_ratio_weight;
        bool filter_solutions;
        bool take_maximum;
        bool support_from_raw_edges;


    cdef struct Detector3DProperties:
        int edge_number
        int strikes
        double center_weight_initial
        double center_weight_final
        vector[int] iteration_numbers
        double residuals_averaged_fraction
        double outlier_factor
        int start_remove_number
        double cauchy_loss_scale
        double eyeball_radius
        double cornea_radius
        double iris_radius
        double n_ref
        refraction_mode run_mode
        vector[int] pars_to_optimize

    cdef enum refraction_mode:
        SWIRSKI, REFRACTION, REFRACTION_APPROXIMATE


cdef extern from 'detect_2d.hpp':


  cdef cppclass Detector2D:

    Detector2D() except +
    shared_ptr[Detector2DResult] detect( Detector2DProperties& prop, Mat& image, Mat& color_image, Mat& debug_image, Rect_[int]& roi, bint visualize , bint use_debug_image)
    shared_ptr[Detector2DResult] detect_legacy( Detector2DProperties& prop, Mat& image, Mat& color_image, Mat& debug_image, Rect_[int]& roi, bint visualize , bint use_debug_image)
    shared_ptr[Detector2DResult] empty_result()

cdef extern from "singleeyefitter/EyeModel.h" namespace "singleeyefitter":


    cdef cppclass EyeModel:

        cppclass PupilParams:
            float theta
            float psi
            float radius

        cppclass Observation:
            shared_ptr[const Detector2DResult] mObservation2D;
            pair[Circle, Circle] mUnprojectedCirclePair
            Observation( shared_ptr[const Detector2DResult] observation, double focalLength)

        EyeModel(int modelId, double timestamp, double focalLength)

        Detector3DResult predictAndUpdate( shared_ptr[Detector2DResult]& results, const Detector3DProperties& prop, bint fillDebugResult )

        int addObservation(shared_ptr[Detector2DResult]& results, int prepare_toggle)
        Detector3DResultRefraction optimize(bool initialization_toggle, const Detector3DProperties&  props)
        void setSphereCenter(vector[double] sphere_center, const Detector3DProperties&  props)
        Detector3DResult predictSingleObservation(shared_ptr[Detector2DResult]& results, bool prepare, const Detector3DProperties&)
        void setApproximationParameters(vector[double], vector[double], vector[double], vector[double], vector[double])

        void reset()
        double getFocalLength()
        Sphere[double] getSphere()

        double mFocalLength
        Sphere[double] mCurrentSphere





