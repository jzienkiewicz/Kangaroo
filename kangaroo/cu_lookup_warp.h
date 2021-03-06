#pragma once

#include "Mat.h"
#include "Image.h"
#include "ImageIntrinsics.h"

namespace roo
{

void CreateMatlabLookupTable(Image<float2> lookup,
    ImageIntrinsics K, float k1, float k2
);

void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2
);

void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2, Mat<float,9> H_no
);

//////////////////////////////////////////////////////

void Warp(
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
);

void Warp(
    Image<float> out, const Image<float> in, const Image<float2> lookup
);

}
