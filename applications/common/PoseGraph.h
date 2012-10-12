#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <boost/ptr_container/ptr_vector.hpp>

#include <sophus/se3.h>
#include "CeresQuatXYZW.h"

double* pt(Sophus::SE3& T) {
    return T.translation().data();
}

double* pq(Sophus::SE3& T) {
    return T.so3().unit_quaternion().coeffs().data();
}

const double* pt(const Sophus::SE3& T) {
    return T.translation().data();
}

const double* pq(const Sophus::SE3& T) {
    return T.so3().unit_quaternion().coeffs().data();
}

// translation, quaternion (x,y,z,  w, i, j, k)
struct Keyframe {
    Keyframe() {
    }

    Keyframe(Sophus::SE3 T_wk)
        : m_T_wk(T_wk)
    {
    }

    Sophus::SE3 GetT_wk() const {
        return m_T_wk;
    }

    void SetT_wk(Sophus::SE3 T_wk) {
        m_T_wk = T_wk;
    }

    Sophus::SE3 m_T_wk;
};

// Indirect measurement of T_wk through T_wz given frame transform T_zk
struct UnaryEdgeIndirect6DofCostFunction
{
    UnaryEdgeIndirect6DofCostFunction( Sophus::SE3 T_wz )
        : m_T_wz(T_wz)
    {
    }

    template <typename T>
    bool operator()(const T* const R_kz, const T* const t_kz, const T* const R_wk, const T* const t_wk, T* residuals) const
    {
        const double* _R_wz = pq(m_T_wz);
        const double* _t_wz = pt(m_T_wz);
        const T R_measured_wz[4] = {(T)_R_wz[0],(T)_R_wz[1],(T)_R_wz[2],(T)_R_wz[3] };
        const T t_measured_wz[3] = {(T)_t_wz[0],(T)_t_wz[1],(T)_t_wz[2] };

        T R_measured_kw[4];
        T t_measured_kw[3];
        XYZUnitQuatXYZWComposeInverse(R_kz, t_kz, R_measured_wz, t_measured_wz, R_measured_kw, t_measured_kw );

        T R_w_mw[4]; T t_w_mw[3];
        XYZUnitQuatXYZWCompose(R_wk, t_wk, R_measured_kw, t_measured_kw, R_w_mw, t_w_mw);

        residuals[0] = t_w_mw[0];
        residuals[1] = t_w_mw[1];
        residuals[2] = t_w_mw[2];
        QuatXYZWToAngleAxis(R_w_mw, residuals+3);
        return true;
    }

    Sophus::SE3 m_T_wz;
};

struct UnaryEdgeXYCostFunction
{
    UnaryEdgeXYCostFunction( Eigen::Vector3d xy )
        : mk_w(xy)
    {
    }

    template <typename T>
    bool operator()(const T* const t_wa, T* residuals) const
    {
        residuals[0] = (T)1E-2 * ((T)mk_w(0) - t_wa[0]);
        residuals[1] = (T)1E-2 * ((T)mk_w(1) - t_wa[1]);
        residuals[2] = (T)1E-2 * ((T)mk_w(2) - t_wa[2]);
        return true;
    }

    Eigen::Vector3d mk_w;
};

struct BinaryEdgeXYZQuatCostFunction
//    :   ceres::SizedCostFunction<7,7>
{
    BinaryEdgeXYZQuatCostFunction( Sophus::SE3 T_ba)
        : m_T_ba(T_ba)
    {
    }

    template <typename T>
    bool operator()(const T* const R_wb, const T* const t_wb, const T* const R_wa, const T* const t_wa, T* residuals) const
    {
        const double* _R_ba = pq(m_T_ba);
        const double* _t_ba = pt(m_T_ba);
        const T R_measured_ba[4] = {(T)_R_ba[0],(T)_R_ba[1],(T)_R_ba[2],(T)_R_ba[3] };
        const T t_measured_ba[3] = {(T)_t_ba[0],(T)_t_ba[1],(T)_t_ba[2] };

        T R_ab[4]; T t_ab[3];
        XYZUnitQuatXYZWInverseCompose(R_wa, t_wa, R_wb, t_wb, R_ab, t_ab);

        T R_a_ma[4]; T t_a_ma[3];
        XYZUnitQuatXYZWCompose(R_ab, t_ab, R_measured_ba, t_measured_ba, R_a_ma, t_a_ma);

        residuals[0] = t_a_ma[0];
        residuals[1] = t_a_ma[1];
        residuals[2] = t_a_ma[2];
        QuatXYZWToAngleAxis(R_a_ma, residuals+3);
        return true;
    }

    // Observed transformation
    Sophus::SE3 m_T_ba;
};

class PoseGraph {
public:
    PoseGraph() {
        quat_param = new QuatXYZWParameterization;
    }

    int AddKeyframe(Keyframe* kf)
    {
        const int id = keyframes.size();
        keyframes.push_back( kf );
        problem.AddParameterBlock(pq(kf->m_T_wk), 4, quat_param);
        problem.AddParameterBlock(pt(kf->m_T_wk), 3, NULL);
        return id;
    }

    int AddKeyframe()
    {
        return AddKeyframe(new Keyframe() );
    }

    Keyframe& GetKeyframe(int a)
    {
        assert(a < keyframes.size());
        return keyframes[a];
    }

    int AddSecondaryCoordinateFrame(Sophus::SE3 T_kz = Sophus::SE3() )
    {
        Keyframe* kf = new Keyframe(T_kz);
        const int id = coord_frames.size();
        coord_frames.push_back( kf );
        problem.AddParameterBlock(pq(kf->m_T_wk), 4, quat_param);
        problem.AddParameterBlock(pt(kf->m_T_wk), 3, NULL);
        return id;
    }

    Keyframe& GetSecondaryCoordinateFrame(int z)
    {
        assert(z < coord_frames.size());
        return coord_frames[z];
    }

    void AddBinaryEdge(int b, int a, Sophus::SE3 T_ba)
    {
        Keyframe& kfa = GetKeyframe(a);
        Keyframe& kfb = GetKeyframe(b);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BinaryEdgeXYZQuatCostFunction, 6, 4, 3, 4, 3>(
                new BinaryEdgeXYZQuatCostFunction(T_ba)
            ), NULL,
            pq(kfb.m_T_wk), pt(kfb.m_T_wk),
            pq(kfa.m_T_wk), pt(kfa.m_T_wk)
        );
    }

    void AddUnaryEdge(int a, Eigen::Vector3d xyz)
    {
        Keyframe& kfa = GetKeyframe(a);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<UnaryEdgeXYCostFunction, 3, 3>(
                new UnaryEdgeXYCostFunction(xyz)
            ), NULL,
            pt(kfa.m_T_wk)
        );
    }

    int AddRelativeKeyframe(int keyframe_a, Sophus::SE3 T_ak)
    {
        const int k = AddKeyframe();
        Keyframe& kf_a = GetKeyframe(keyframe_a);
        Keyframe& kf_k = GetKeyframe(k);

        // Initialise keyframes pose based on relative transform
        kf_k.SetT_wk( kf_a.GetT_wk() * T_ak );
//        kf_k.SetT_wk(kf_a.GetT_wk() * Sophus::SE3(Sophus::SO3(), Vector3d(0.1,0,0)));

        AddBinaryEdge(keyframe_a,k,T_ak);
        return k;
    }

    void AddIndirectUnaryEdge(int kf_a, int coord_z, Sophus::SE3 T_wz)
    {
        Keyframe& coz = GetSecondaryCoordinateFrame(coord_z);
        Keyframe& kfa = GetKeyframe(kf_a);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<UnaryEdgeIndirect6DofCostFunction, 6, 4, 3, 4, 3>(
                new UnaryEdgeIndirect6DofCostFunction(T_wz)
            ), NULL,
            pq(coz.m_T_wk), pt(coz.m_T_wk),
            pq(kfa.m_T_wk), pt(kfa.m_T_wk)
        );

//        // Only optimise rotation
//        problem.SetParameterBlockConstant( pt(coz.m_T_wk) );
    }

    void Solve()
    {
        ceres::Solver::Options options;
//        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 4;
//        options.check_gradients = true;
        options.update_state_every_iteration = true;
        options.max_num_iterations = 1000;
//        options.gradient_tolerance = 1E-50;
//        options.function_tolerance = 1E-50;
//        options.parameter_tolerance = 1E-50;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;
    }

//protected:
    ceres::LocalParameterization* quat_param;
    boost::ptr_vector<Keyframe> keyframes;
    boost::ptr_vector<Keyframe> coord_frames;
    ceres::Problem problem;
};
