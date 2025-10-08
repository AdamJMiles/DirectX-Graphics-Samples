#include "Shared.h"


StructuredBuffer<NetworkOffsets> g_networkOffsets : register(t3);

ByteAddressBuffer g_imageBuffer : register(t0);
ByteAddressBuffer g_labelBuffer : register(t1);

RWByteAddressBuffer g_debugBuffer : register(u2);

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

float16_t ActivationFunctionDerivative(float16_t x)
{
    return x * (1.0h - x);
}

template<int N>
vector<float16_t, N> ActivationFunctionDerivative(inout vector<float16_t, N> input)
{
    vector<float16_t, N> output;

    for (uint i = 0; i < N; ++i)
    {
        output[i] = ActivationFunctionDerivative(input[i]);
    }

    return output;
}