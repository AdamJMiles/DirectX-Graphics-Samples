#include "Header.hlsli"

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer g_resultsBuffer : register(u1);

[RootSignature(RootSig)]
[numthreads(14, 14, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint2 gid : SV_GroupThreadID, uint2 groupID : SV_GroupID)
{
	RWTexture2D<float4> gOutput = ResourceDescriptorHeap[0];

	uint width, height;
	gOutput.GetDimensions(width, height);

	uint imagesPerRow = 150;//width / 28;

	uint2 imageCoord = groupID;
	uint imageIndex = imageCoord.y * imagesPerRow + imageCoord.x;

	if (imageIndex >= 10000)
	{
		gOutput[DTid.xy] = float4(0, 0, 1, 1);
		return;
	}

	uint pixelIndex = (gid.y * 2) * 28 + (gid.x * 2);
	float16_t pixel = GetPixel(imageIndex, pixelIndex);
	float16_t3 color = pixel.xxx;

	vector<float16_t, 12> results = g_resultsBuffer.Load<vector<float16_t, 12> >(imageIndex * sizeof(vector<float16_t, 12>));
	bool isCorrect = (results[0] == results[1]);

	if (pixel == 0.0f)
	{
		if (isCorrect)
			color = float16_t3(0, 1, 0);
		else
			color = float16_t3(1, 0, 0);
	}

	gOutput[DTid.xy] = float4(color, 1);
}