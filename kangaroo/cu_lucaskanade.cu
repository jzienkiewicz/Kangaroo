#include "cu_lucaskanade.h"

#include <kangaroo/LeastSquareSum.h>
#include <kangaroo/launch_utils.h>

#include <kangaroo/MatUtils.h>

namespace roo
{
__host__ __device__ inline roo::Mat<float, 3, 3> SE2gen(unsigned int genIdx)
{
    roo::Mat<float, 3, 3> gen;
    gen.SetZero();
    switch (genIdx)
    {
    case 0:
        gen(0, 2) = 1;
        break;
    case 1:
        gen(1, 2) = 1;
        break;
    case 2:
        gen(0, 1) = -1;
        gen(1, 0) =  1;
        break;
    }
    return gen;
}

template<typename TO, typename TI>
__global__ void KernWarp(roo::Image<TO> dOutput,
                         roo::Image<TI> dInput,
                         roo::Mat<float, 3, 3> H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float3 po_h = make_float3(x, y, 1);
    float3 pi_h = mulSO3(H, po_h);

    float2 po = dn(po_h);
    float2 pi = dn(pi_h);

    if (dInput.InBounds(pi) && dOutput.InBounds(po))
    {
        TI pix = dInput.template GetBilinear<TI>(pi);
        dOutput(po.x, po.y) = ConvertPixel<TO,TI>(pix);
    }
    else
    {
        dOutput(po.x, po.y) = roo::ConvertPixel<TO, float>(0.0f);
    }
}

template<typename TO, typename TI>
void ImageWarp(roo::Image<TO> d_ouput,
               roo::Image<TI> d_input,
               roo::Mat<float, 3, 3> H)
{
    dim3 block;
    dim3 grid;
    roo::InitDimFromOutputImageOver(block, grid, d_ouput, 16, 16);

    KernWarp<TO,TI> <<< grid, block>>>(d_ouput, d_input, H);
    GpuCheckErrors();
}

template<typename TO, typename TI>
__global__ void KernWarp(roo::Image<TO> dOutput,
                         roo::Image<TI> dInput,
                         roo::Image<float> dDepth,
                         roo::Mat<float, 3, 4> T,
                         roo::ImageIntrinsics K)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float d = dDepth(x, y);

    float2 po   = make_float2(x, y);
    float3 pi_h = roo::mulSE3(T, K.Unproject(po, d));
    float2 pi   = K.Project(pi_h);

    if (dInput.InBounds(pi) && dOutput.InBounds(po))
    {
        TI pix = dInput.template GetBilinear<TI>(pi);
        dOutput(po.x, po.y) = ConvertPixel<TO,TI>(pix);
    }
    else
    {
        dOutput(po.x, po.y) = roo::ConvertPixel<TO, float>(0.0f);
    }
}

template<typename TO, typename TI>
void ImageWarp(roo::Image<TO> d_ouput,
               roo::Image<TI> d_input,
               roo::Image<float> d_depth,
               roo::Mat<float, 3, 4> T,
               roo::ImageIntrinsics K)
{
    dim3 block;
    dim3 grid;
    roo::InitDimFromOutputImageOver(block, grid, d_ouput, 16, 16);

    KernWarp<TO,TI> <<< grid, block>>>(d_ouput, d_input, d_depth, T, K);
    GpuCheckErrors();
}

template<typename TO, unsigned N>
__global__ void KernLucasKanade(roo::Image<TO> dRef,
                                roo::Image<TO> dTemp,
                                roo::Mat<float, 3, 3> H,
                                roo::Image<roo::LeastSquaresSystem<float, N> > dSum)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float3 pt_h = make_float3(x, y, 1);
    float3 pr_h = roo::mulSO3(H, pt_h);

    float2 pt = roo::dn(pt_h);
    float2 pr = roo::dn(pr_h);

    __shared__ roo::SumLeastSquaresSystem<float, 3, 16, 16> shared_lss;
    roo::LeastSquaresSystem<float, N>& local_lss = shared_lss.ZeroThisObs();

    if (dRef.InBounds(pr, 1) && dTemp.InBounds(pt, 1))
    {
        TO pixRef  = dRef.template GetBilinear<TO>(pr);
        TO pixTemp = dTemp.template GetBilinear<TO>(pt);
        TO pixErr  = pixRef - pixTemp;

        const roo::Mat<float, 2, 3> dPr_by_dpr =
        {
            1.0 / pr_h.z,  0.0,        -pr_h.x / (pr_h.z* pr_h.z),
            0.0,         1.0 / pr_h.z, -pr_h.y / (pr_h.z* pr_h.z)
        };

        roo::Mat<TO, 1, 2>    dIr = dRef.template GetCentralDiff<TO>(pr);
        // from here ignore TO, do type convertion and work on float
        roo::Mat<float, 1, 2> dIr_f;

        dIr_f(0, 0) = roo::ConvertPixel<float, TO>(dIr(0, 0));
        dIr_f(0, 1) = roo::ConvertPixel<float, TO>(dIr(0, 1));

        roo::Mat<float, N, 1> Jr;
        for (int i = 0; i < N; i++)
        {
            Jr(i, 0) = dIr_f * dPr_by_dpr * SE2gen(i) * roo::make_mat(pt_h);
        }

        float err = roo::ConvertPixel<float, TO>(pixErr);
        float w = 1;
        local_lss.JTJ   = roo::OuterProduct(Jr, w);
        local_lss.JTy   = Jr * err * w;
        local_lss.obs   = 1;
        local_lss.sqErr = err * err;
    }

    shared_lss.ReducePutBlock(dSum);
}

template<typename TO>
roo::LeastSquaresSystem<float, 3> LucasKanade(roo::Image<TO> d_reference, roo::Image<TO> d_template, roo::Image<unsigned char> d_workspace, roo::Mat<float, 3, 3> H)
{
    dim3 block;
    dim3 grid;
    roo::InitDimFromOutputImageOver(block, grid, d_template, 16, 16);

    roo::HostSumLeastSquaresSystem<float, 3> global_lss(d_workspace, block, grid);

    KernLucasKanade<TO, 3> <<< grid, block>>>(d_reference, d_template, H, global_lss.LeastSquareImage());
    GpuCheckErrors();
    roo::LeastSquaresSystem<float, 3> lss = global_lss.FinalSystem();

    return lss;
}

// templates instantations
// uchars are not supported as there is no function lerp for them;
// lerp is a basic linear interpolation function required for GetBilinear;
template void ImageWarp<float4,float4>(roo::Image<float4> d_ouput, roo::Image<float4> d_input, roo::Mat<float, 3, 3> H);
template void ImageWarp<float,float>(roo::Image<float> d_ouput, roo::Image<float> d_input, roo::Mat<float, 3, 3> H);
template void ImageWarp<float,float>(roo::Image<float> d_ouput, roo::Image<float> d_input, roo::Image<float> d_depth, roo::Mat<float, 3, 4> T, roo::ImageIntrinsics K);
template void ImageWarp<float,unsigned char>(roo::Image<float> d_ouput, roo::Image<unsigned char> d_input, roo::Image<float> d_depth, roo::Mat<float, 3, 4> T, roo::ImageIntrinsics K);
template void ImageWarp<float,unsigned char>(roo::Image<float> d_ouput, roo::Image<unsigned char> d_input, roo::Mat<float, 3, 3> H);

template roo::LeastSquaresSystem<float, 3> LucasKanade<float>(roo::Image<float> d_reference, roo::Image<float> d_template, roo::Image<unsigned char> d_workspace, roo::Mat<float, 3, 3> H);
template roo::LeastSquaresSystem<float, 3> LucasKanade<float4>(roo::Image<float4> d_reference, roo::Image<float4> d_template, roo::Image<unsigned char> d_workspace, roo::Mat<float, 3, 3> H);
}
