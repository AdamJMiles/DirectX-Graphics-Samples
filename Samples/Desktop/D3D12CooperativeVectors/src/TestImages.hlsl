#include "Header.hlsli"

#include <dx/linalg.h>
using namespace dx::linalg;

cbuffer Params
{
    uint numImages;
};

ByteAddressBuffer networkInputs : register(t2);
RWByteAddressBuffer g_resultsBuffer : register(u1);
RWByteAddressBuffer g_epochBuffer : register(u3);

[numthreads(32, 1, 1)]
[RootSignature(RootSig)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint imageIndex = DTid.x;

    if (imageIndex >= numImages)
        return;

    vector<float16_t, NUM_INPUT_NEURONS> inputPixels = GetPixels<NUM_INPUT_NEURONS>(imageIndex, 0);

    NetworkOffsets layer0 = g_networkOffsets[0];
    NetworkOffsets layer1 = g_networkOffsets[1];

    MatrixRef<DATA_TYPE_FLOAT16, NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> HiddenLayerWeights = { networkInputs, layer0.weightsOffset, 0 };
    VectorRef<DATA_TYPE_FLOAT16> HiddenLayerBiases = { networkInputs, layer0.biasesOffset };

    vector<float16_t, NUM_HIDDEN_NEURONS> hiddenLayerOutput = MulAdd<float16_t>(HiddenLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(inputPixels), HiddenLayerBiases);
    ActivationFunction(hiddenLayerOutput);

    MatrixRef<DATA_TYPE_FLOAT16, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> OutputLayerWeights = { networkInputs, layer1.weightsOffset, 0 };
    MatrixRef<DATA_TYPE_FLOAT16, NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL, true> OutputLayerWeightsTransposed = { networkInputs, layer1.weightsOffset, 0 };
    VectorRef<DATA_TYPE_FLOAT16> OutputLayerBiases = { networkInputs, layer1.biasesOffset };

    vector<float16_t, NUM_OUTPUT_NEURONS> outputLayerOutput = MulAdd<float16_t>(OutputLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(hiddenLayerOutput), OutputLayerBiases);
    ActivationFunction(outputLayerOutput);

    // Figure out the index of the max output neuron
    uint maxIndex = 0;
    float16_t maxValue = outputLayerOutput[0];
    for (uint i = 1; i < NUM_OUTPUT_NEURONS; i++)
    {
        if (outputLayerOutput[i] > maxValue)
        {
            maxValue = outputLayerOutput[i];
            maxIndex = i;
        }
    }

    uint correctLabel = GetLabel(imageIndex);

    // Store the result (predicted class) in the results buffer
    vector<float16_t, 12> results;
    results[0] = (float16_t)correctLabel;
	results[1] = (float16_t)maxIndex;

    for (uint i = 0; i < NUM_OUTPUT_NEURONS; i++)
		results[i + 2] = outputLayerOutput[i];

    g_resultsBuffer.Store(imageIndex * sizeof(results), results);

    if(correctLabel == maxIndex)
    {
        g_epochBuffer.InterlockedAdd(0, 1);
	}
}