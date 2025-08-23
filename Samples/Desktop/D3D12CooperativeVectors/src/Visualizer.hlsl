#include "Header.hlsli"

#include <dx/linalg.h>
using namespace dx::linalg;

ByteAddressBuffer inputWeights : register(t2);
ByteAddressBuffer inputBiases : register(t3);

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

    vector<float16_t, 1> inputVector = { pixel };
    VectorRef<DATA_TYPE_FLOAT16> Biases = {inputBiases, 0 };
    MatrixRef<DATA_TYPE_FLOAT16, 1, 1, MATRIX_LAYOUT_MUL_OPTIMAL> MulMatrix = { inputWeights, 0, 0 };

	gOutput[DTid.xy] = float4(pixel.xxx, 1);
}