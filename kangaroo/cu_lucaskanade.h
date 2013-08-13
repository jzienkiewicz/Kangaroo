#pragma once

#include <kangaroo/Image.h>
#include <kangaroo/ImageIntrinsics.h>
#include <kangaroo/Mat.h>

namespace roo
{

template<typename TO>
roo::LeastSquaresSystem<float, 3> LucasKanade(roo::Image<TO> d_reference,
                                              roo::Image<TO> d_template,
                                              roo::Image<unsigned char> d_workspace,
                                              roo::Mat<float, 3, 3> H);

template<typename TO, typename TI>
void ImageWarp(roo::Image<TO> d_ouput,
               roo::Image<TI> d_input,
               roo::Mat<float, 3, 3> H);

template<typename TO, typename TI>
void ImageWarp(roo::Image<TO> d_ouput,
               roo::Image<TI> d_input,
               roo::Image<float> d_depth,
               roo::Mat<float, 3, 4> T,
               roo::ImageIntrinsics K);

}
