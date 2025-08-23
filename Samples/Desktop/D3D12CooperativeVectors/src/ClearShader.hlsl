#include "Header.hlsli"

RWByteAddressBuffer output : register(u0);

cbuffer Params
{
    uint numFP16ValuesToClear;
};

[RootSignature(RootSig)]
[numthreads(32, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    if(DTid.x < numFP16ValuesToClear)
    {
        uint address = DTid.x * 2; // Each float16_t is 2 bytes
        output.Store<float16_t>(address, (float16_t)0.0f);
    }
}