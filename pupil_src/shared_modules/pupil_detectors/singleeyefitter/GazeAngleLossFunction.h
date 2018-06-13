#ifndef GAZEANGLELOSSFUNCTION_H__
#define GAZEANGLELOSSFUNCTION_H__

#include "projection.h"
#include "geometry/Sphere.h"
#include "geometry/Ellipse.h"

namespace singleeyefitter{

    template<typename Scalar>
    class GazeAngleLossFunctionVectorized{

        public:

            GazeAngleLossFunctionVectorized(const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& xy,
                                                const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& gv1,
                                                const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& gv2,
                                                const Scalar eye_radius,
                                                const Scalar focal_length,
                                                const Scalar CauchyLossScale):
            xy(xy), gv1(gv1), gv2(gv2), eye_radius(eye_radius), focal_length(focal_length), cls(CauchyLossScale){

                N = xy.cols();

                cls = 1./pow(CauchyLossScale,2);

                l.resize(3,N);
                l.row(0) = (xy.row(0).array()-320.)/focal_length;
                l.row(1) = (xy.row(1).array()-240.)/focal_length;
                l.row(2).setZero();
                l.row(2) = l.row(2).array()+1.0;
                l.colwise().normalize();

            }

            template <typename T>
            bool operator()(const T* const sphere_center, T* e) const
            {

                Eigen::Matrix<T,3,Eigen::Dynamic> l_ = l.template cast<T>();
                Eigen::Matrix<T,3,1> s = {sphere_center[0],sphere_center[1],sphere_center[2]};
                Eigen::Matrix<T,Eigen::Dynamic,1> temp = l_.transpose()*(-s);
                Eigen::Matrix<T,Eigen::Dynamic,1> Delta = Eigen::square(temp.array())-T(pow(s.norm(),2))+T(pow(eye_radius,2));
                Eigen::Matrix<T,3,Eigen::Dynamic,1> P;
                P.resize(3,N);

                for (int i=0;i<N;i++){
                    if (Delta(i)<0.0){
                        P.col(i) = l_.col(i)*(-temp(i,0));                      //Closest point on gaze ray to sphere center
                    }else{
                        P.col(i) = (-temp(i,0)-sqrt(Delta(i,0)))*l_.col(i);     //Intersection of gaze ray with sphere center
                    }
                }

                Eigen::Matrix<T,3,Eigen::Dynamic> n = P.colwise()-s;
                n.colwise().normalize();

                temp = (gv1.template cast<T>().transpose()*n).diagonal().array().max((gv2.template cast<T>().transpose()*n).diagonal().array());
                temp = sqrt(T(cls)*log(T(1.)+T(cls)*Eigen::square(acos(temp.array()))));

                for (int i=0;i<N;i++){

                    e[i] = temp(i);

                }

                return true;

            }

        private:

            int N;
            const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& xy;
            const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& gv1;
            const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& gv2;
            Eigen::Matrix<Scalar, 3, Eigen::Dynamic> l;
            const Scalar eye_radius;
            const Scalar focal_length;
            Scalar cls;

    };

} // namespace singleeyefitter

#endif /* end of include guard: GAZEANGLELOSSFUNCTION_H__ */
