#ifndef REFRACTIONRESIDUALFUNCTION_H__
#define REFRACTIONRESIDUALFUNCTION_H__

#include "projection.h"
#include "geometry/Sphere.h"
#include "geometry/Ellipse.h"
#include "utils.h"
#include "mathHelper.h"
#include <fstream>
#include <random>
#include <algorithm>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

namespace singleeyefitter{

template<typename T>
Eigen::Matrix<T, 3, 1> map_to_tangent_space(const cv::Point& inlier,
                            const T* const eye_center,
                            const T* const eye_param,
                            const T* const pupil_param,
                            double focal_length,
                            double nref_ = 1.376)
{

                typedef Eigen::Matrix<T, 3, 1> Vector3;

                T nref = T(nref_);

                T re = T(eye_param[0]);
                T rc = T(eye_param[1]);
                T ri = T(eye_param[2]);
                T dp = sqrt( pow(re, 2) - pow(ri, 2) ); // mm
                T h  = sqrt( pow(rc, 2) - pow(ri, 2) ); // mm
                T dc = dp - h;

                int type_;

                //GET PARAMETERS
                const Vector3 sphere_center{eye_center[0],eye_center[1],eye_center[2]};

                T theta = pupil_param[0];
                T phi = pupil_param[1];
                T r = pupil_param[2];

                Vector3 par_spherical;
                par_spherical[0] = sin(theta)*cos(phi);
                par_spherical[1] = cos(theta);
                par_spherical[2] = sin(theta)*sin(phi);

                Vector3 p_center;  // PUPIL CENTER
                p_center = sphere_center + dp * par_spherical;

                Vector3 p_normal;  // PUPIL NORMAL
                p_normal = (p_center-sphere_center).normalized();

                //INITIAL RAY
                Vector3 ray_center;
                Vector3 ray_direction;

                ///////
                Vector3 p_intersect;
                Vector3 lcc;
                Vector3 lccn;           //INTERSECTION WITH CORNEA, NORMAL OF CORNEA AT THAT POINT
                Vector3 minuslccn;
                Vector3 vnew;           //REFRACTED RAY

                T norm_;
                T d, d1_cornea, d2_cornea, d1_eyeball, d2_eyeball, d_cornea, d_eyeball;
                T distance;
                Vector3 cornea_center;
                T temp, temp2, temp5;
                Vector3 temp3, temp4;
                T discriminant_cornea, discriminant_eyeball;
                Vector3 a, b, acrossb;
                Vector3 ncrossv, minncrossv, ncrossminncrossv;

                Vector3 upprojected_edge;
                Vector3 tangent_plane_delta;


                //INITIAL RAY
                ray_center << T(0.0), T(0.0), T(0.0);

                ray_direction[0] = T(inlier.x)  - ray_center[0];
                ray_direction[1] = T(inlier.y)  - ray_center[1];
                ray_direction[2] = T(focal_length) - ray_center[2];
                ray_direction = ray_direction.normalized();

                //SETUP CORNEA CENTER
                cornea_center = sphere_center+dc*p_normal;

                //CHECK FOR INTERSECTION WITH CORNEA
                temp =  ray_direction.dot(ray_center-cornea_center);
                discriminant_cornea  = pow(temp,2);
                discriminant_cornea -= pow((ray_center-cornea_center).norm(),2);
                discriminant_cornea += pow(rc,2);

                //CHECK FOR INTERSECTION WITH EYEBALL
                temp2 =  ray_direction.dot(ray_center-sphere_center);
                discriminant_eyeball  = pow(temp2,2);
                discriminant_eyeball -= pow((ray_center-sphere_center).norm(),2);
                discriminant_eyeball += pow(re,2);

                if(discriminant_cornea<0.0 && discriminant_eyeball>0.0){  //INTERSECTION WITH EYEBALL BUT NOT WITH CORNEA

                         d1_eyeball = -temp2 + sqrt(discriminant_eyeball);
                         d2_eyeball = -temp2 - sqrt(discriminant_eyeball);

                         d = ( d1_eyeball < d2_eyeball ) ? d1_eyeball : d2_eyeball;

                         type_ = 2;

                }

                if(discriminant_cornea>0.0 && discriminant_eyeball>0.0){  //INTERSECTION WITH EYEBALL AND WITH CORNEA

                         d1_cornea = -temp + sqrt(discriminant_cornea);
                         d2_cornea = -temp - sqrt(discriminant_cornea);
                         d_cornea = ( d1_cornea < d2_cornea ) ? d1_cornea : d2_cornea;

                         d1_eyeball = -temp2 + sqrt(discriminant_eyeball);
                         d2_eyeball = -temp2 - sqrt(discriminant_eyeball);
                         d_eyeball = ( d1_eyeball < d2_eyeball ) ? d1_eyeball : d2_eyeball;

                         d = ( d_cornea < d_eyeball ) ? d_cornea : d_eyeball;

                         type_ = ( d_cornea < d_eyeball ) ? 1 : 2;

                }

                if(discriminant_cornea>0.0 && discriminant_eyeball<0.0){  //INTERSECTION WITH CORNEA BUT NOT WITH EYEBALL

                         d1_cornea = -temp + sqrt(discriminant_cornea);
                         d2_cornea = -temp - sqrt(discriminant_cornea);
                         d = ( d1_cornea < d2_cornea ) ? d1_cornea : d2_cornea;

                         type_ = 3;

                }

                if(discriminant_cornea<0.0 && discriminant_eyeball<0.0){  //INTERSECTION WITH NEITHER CORNEA NOR EYEBALL

                        type_ = 0;

                }

                switch ( type_ ){

                    case 2: //INTERSECTION WITH EYEBALL

                        lcc = ray_center + d * ray_direction;
                        distance = (p_center-lcc).norm();

                        upprojected_edge  = lcc  - ( lcc - p_center ).dot( p_normal )*p_normal;
                        tangent_plane_delta = upprojected_edge - p_center;
                        upprojected_edge = p_center + distance*tangent_plane_delta.normalized();

                        break;

                    case 1: //INTERSECTION WITH CORNEA (+EYEBALL)-> REFRACTING RAY
                    case 3: //INTERSECTION WITH CORNEA (-EYEBALL)-> REFRACTING RAY

                        //CORNEAL INTERSECTION
                        lcc = ray_center + d * ray_direction;

                        //NORMAL AT CORNEAL INTERSECTION
                        lccn = (lcc - cornea_center).normalized();

                        //REFRACT RAY
                        minuslccn = -lccn;
                        ncrossv = lccn.cross(ray_direction);
                        minncrossv = minuslccn.cross(ray_direction);
                        ncrossminncrossv = lccn.cross(minncrossv);

                        //WRITE TO OLD VARIABLES
                        ray_center = lcc;

                        ray_direction = 1.0/nref*ncrossminncrossv-lccn*sqrt(1.0-1.0/pow(nref,2)*pow(ncrossv.norm(),2));

                        //INTERSECT FINAL RAY WITH TANGENT PLANE
                        d  = (p_center-ray_center).dot(p_normal);
                        d /= ray_direction.dot(p_normal);

                        p_intersect = ray_center+ d * ray_direction;
                        upprojected_edge = p_intersect;

                        break;

                    case 0: //NEAREST POINT ON EYEBALL

                     //CLOSEST POINT TO EYEBALL
                        temp3 = sphere_center;
                        //temp3 = cornea_center;
                        temp5 = ray_direction.dot(temp3);
                        temp4 = ray_center + temp5 * ray_direction; // CLOSEST POINT ON RAY

                        distance = (temp4-sphere_center).norm();
                        //distance = (temp4-cornea_center).norm();

                        T p = - p_normal.dot(p_center);

                        Vector3 normal = (ray_direction.cross(sphere_center)).normalized();
                        //Vector3 normal = (ray_direction.cross(cornea_center)).normalized();
                        Vector3 point;
                        point[0] = T(0.0);
                        point[1] = T(0.0);
                        point[2] = T(0.0);

                        Vector3 a = p_normal.cross(normal);
                        Vector3 x0;
                        x0[0] =  p * normal[1]/(p_normal[1]*normal[0]-normal[1]*p_normal[0]);
                        x0[1] = -p * normal[0]/(p_normal[1]*normal[0]-normal[1]*p_normal[0]);
                        x0[2] = T(0.0);

                        Vector3 b = x0-sphere_center;
                        //Vector3 b = x0-cornea_center;

                        T A = pow(a.norm(),2);
                        T B = 2.0*a.dot(b);
                        T C = pow(b.norm(),2);

                        if( pow( B, 2 ) / ( 4.0*pow(A,2) ) - (C-pow(distance,2))/A < 0.0 ){
                            //std::cout<<"Case 0 is failing!"<<std::endl;
                            upprojected_edge = temp4;
                            break;
                        }

                        T t1 = -B/(2.0*A) + sqrt( pow( B, 2 ) / ( 4.0*pow(A,2) ) - (C-pow(distance,2))/A);
                        T t2 = -B/(2.0*A) - sqrt( pow( B,2 ) / ( 4.0*pow(A,2) ) - (C-pow(distance,2))/A);


                        Vector3 p1 = x0+t1*a;
                        Vector3 p2 = x0+t2*a;

                        if (p1[2]>p2[2]){
                            upprojected_edge = p1;
                        }else{
                            upprojected_edge = p2;
                        }

//                        //CLOSEST POINT TO EYEBALL
//                        temp3 = sphere_center;
//                        temp5 = ray_direction.dot(temp3);
//                        temp4 = ray_center + temp5 * ray_direction; // CLOSEST POINT ON RAY

//                        //CLOSEST POINT ON EYEBALL
//                        temp3 = (temp4 - sphere_center).normalized();
//                        lcc = sphere_center + re * temp3; // CLOSEST POINT ON EYEBALL

//                        distance  = (temp4-lcc).norm();
//                        distance += (p_center-lcc).norm();
//
//                        upprojected_edge  = temp4 - ( temp4 - p_center ).dot( p_normal )*p_normal;
//                        tangent_plane_delta = upprojected_edge - p_center;
//                        upprojected_edge = p_center + distance * tangent_plane_delta.normalized();

                        break;

                        }

                return upprojected_edge;
}

template<typename Scalar>
class RefractionResidualFunction
{

   public:

        RefractionResidualFunction(const std::vector<cv::Point>& edges,
                                            const Scalar& eyeball_radius,
                                            const Scalar& focal_length,
                                            const cv::Point ellipse_center,
                                            double * const lambda_2,
                                            const int N):
                                            edges(edges),
                                            eyeball_radius(eyeball_radius),
                                            focal_length(focal_length),
                                            ellipse_center(ellipse_center),
                                            lambda_2(lambda_2),
                                            N(N)
                                            {

                                              for (int i=0; i<edges.size(); i++){
                                                v.push_back(i);
                                              }

                                              std::random_device rd;
                                              std::mt19937 g(rd());
                                              std::shuffle(v.begin(), v.end(), g);

                                              internal_edges = std::vector<cv::Point>();
                                              for (int m = 0; m < N; ++m) {
                                                    internal_edges.push_back(edges[v[m]]);
                                              }

                                            }

        template <typename T>
        bool operator()(const T* const eye_center, const T* eye_param, const T* const pupil_param, T * e) const
        {
            Eigen::Matrix<T, 3, 1> upprojected_edge;
            int i;

            const Eigen::Matrix<T, 3, 1> extended_eye_param{T(eyeball_radius), T(eye_param[0]), T(eye_param[1])};
            const Eigen::Matrix<T, 3, 1> sphere_center{T(eye_center[0]),T(eye_center[1]),T(eye_center[2])};

            T theta = pupil_param[0];
            T phi = pupil_param[1];
            T r = pupil_param[2];
            T dp = sqrt( pow(extended_eye_param[0], 2) - pow(extended_eye_param[2], 2) );

            Eigen::Matrix<T, 3, 1> par_spherical;
            par_spherical[0] = sin(theta)*cos(phi);
            par_spherical[1] = cos(theta);
            par_spherical[2] = sin(theta)*sin(phi);

            Eigen::Matrix<T, 3, 1> p_center;
            p_center = sphere_center + dp * par_spherical;

            for (int i = 0; i < N; ++i) {
                upprojected_edge = map_to_tangent_space<T>(internal_edges[i], &sphere_center[0], &extended_eye_param[0], pupil_param, focal_length);
                e[i] = (r - (p_center-upprojected_edge).norm())/sphere_center.norm();

            }

            upprojected_edge = map_to_tangent_space<T>(ellipse_center, &sphere_center[0], &extended_eye_param[0], pupil_param, focal_length);
            e[N] = T(*lambda_2)*(p_center-upprojected_edge).norm()/sphere_center.norm();

            return true;
        }

    private:

        const std::vector<cv::Point>& edges;
        std::vector<cv::Point> internal_edges;
        const Scalar& eyeball_radius;
        const Scalar& focal_length;
        std::string filename;
        const cv::Point ellipse_center;
        double * const lambda_2;
        std::vector<int> v;
        const int N;

};

} // namespace singleeyefitter

#endif /* end of include guard: REFRACTIONRESIDUALFUNCTION_H__ */