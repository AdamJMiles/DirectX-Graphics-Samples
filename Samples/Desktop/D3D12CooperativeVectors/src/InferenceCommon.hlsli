#include "Header.hlsli"

cbuffer Params
{
    uint weightsOffset;
    uint biasesOffset;
};

[RootSignature(RootSig)]
[numthreads(IMAGE_SIZE_IN_PIXELS, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID)
{  
    uint imageIndex = 0;
    uint laneID = GTid.x;
    
    float16_t inputPixel = GetPixel(imageIndex, laneID);

#if defined(USE_COOPERATIVE_VECTORS)

#else

#endif

    
}