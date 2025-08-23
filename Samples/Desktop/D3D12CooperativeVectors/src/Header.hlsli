#include "Shared.h"

#define RootSig "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(b0, num32BitConstants=3), SRV(t0), SRV(t1), UAV(u0), UAV(u1), SRV(t2), SRV(t3)"

ByteAddressBuffer g_imageBuffer : register(t0);
ByteAddressBuffer g_labelBuffer : register(t1);

float16_t GetPixel(uint imageIndex, uint inputIndex)
{
	uint byteOffsetToImage = imageIndex * IMAGE_SIZE_IN_BYTES;
	uint byteOffsetToPixel = byteOffsetToImage + inputIndex;

	uint alignedOffset = byteOffsetToPixel & ~1;
	uint16_t word = g_imageBuffer.Load<uint16_t>(alignedOffset);

	return ((byteOffsetToPixel & 1) ? word >> 8 : word & 0xFF) / 255.0h;
}

template<int N>
vector<float16_t, N> GetPixels(uint imageIndex, uint startIndex)
{
    // TODO. Optimise for multiples of 4.
    vector<float16_t, N> pixels;
    for (uint i = 0; i < N; ++i)
    {
        pixels[i] = GetPixel(imageIndex, startIndex + i);
    }
    return pixels;
}

uint16_t GetLabel(uint index)
{
	uint byteOffset = index & ~1;
	uint16_t word = g_labelBuffer.Load<uint16_t>(byteOffset);

	return (index & 1) ? word >> 8 : word & 0xFF;
}

template<int N>
void ActivationFunction(inout vector<float16_t, N> x)
{
    for (uint i = 0; i < N; ++i)
    {
        x[i] = 1.0h / (1.0h + exp(-x[i]));
    }
}