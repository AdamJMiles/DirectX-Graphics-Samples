#include "Header.hlsli"

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer debugBuffer : register(u1);

[RootSignature(RootSig)]
[numthreads(28, 28, 1)]
void main( uint3 DTid : SV_DispatchThreadID, uint2 gid : SV_GroupThreadID)
{
	RWTexture2D<float4> gOutput = ResourceDescriptorHeap[0];

	uint width, height;
	gOutput.GetDimensions(width, height);

	uint imagesPerRow = width / 28;

	uint2 imageCoord = DTid.xy / 28;
	uint imageIndex = 0;//imageCoord.y * imagesPerRow + imageCoord.x;
    	
	uint pixelIndex = gid.y * 28 + gid.x;
	float16_t pixel = GetPixel(imageIndex, pixelIndex);

    float16_t firstValueInDebug = debugBuffer.Load<float16_t>(0);

	gOutput[DTid.xy] = float4(pixel.xxx, firstValueInDebug);
}