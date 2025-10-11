#include "Header.hlsli"

#define USE_COOPERATIVE_VECTORS

#if defined(USE_COOPERATIVE_VECTORS)
#include <dx/linalg.h>
using namespace dx::linalg;
#endif
                                             
static const float16_t LEARNING_RATE = 0.05h;
static const uint NUM_INPUTS_TO_LOAD = 784;

cbuffer Params
{
    uint forwardPassStride; // in bytes
    uint numImages;
    uint firstImage;
};

RWByteAddressBuffer networkInputs : register(u0);
RWByteAddressBuffer forwardPassOutput : register(u1);

RWByteAddressBuffer debugBuffer : register(u2);

[RootSignature(RootSig)]
[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{  
    if(DTid.x >= numImages)
        return;

    uint imageIndex = DTid.x + firstImage;

#if defined(USE_COOPERATIVE_VECTORS)
    
    vector<float16_t, NUM_INPUTS_TO_LOAD> inputPixels = GetPixels<NUM_INPUTS_TO_LOAD>(imageIndex, 0);

    // Load the hiddenLayerInput and outLayerInput from the forward pass buffer
    uint inputBase = (imageIndex * forwardPassStride);

    vector<float16_t, NUM_HIDDEN_NEURONS> hiddenLayerOutput = forwardPassOutput.Load<vector<float16_t, NUM_HIDDEN_NEURONS> >(inputBase);
    inputBase += sizeof(float16_t) * NUM_HIDDEN_NEURONS;

    vector<float16_t, NUM_HIDDEN_NEURONS> hiddenErrorSignal = forwardPassOutput.Load<vector<float16_t, NUM_HIDDEN_NEURONS> >(inputBase);
    inputBase += sizeof(float16_t) * NUM_HIDDEN_NEURONS;

    vector<float16_t, NUM_OUTPUT_NEURONS> outputErrorSignal = forwardPassOutput.Load<vector<float16_t, NUM_OUTPUT_NEURONS> >(inputBase);
    
    outputErrorSignal *= LEARNING_RATE;
    hiddenErrorSignal *= LEARNING_RATE;
        
    NetworkOffsets layer0 = g_networkOffsets[0];
    NetworkOffsets layer1 = g_networkOffsets[1];

    RWMatrixRef<DATA_TYPE_FLOAT16, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> Layer1Weights = { networkInputs, layer1.weightsOffset, 0 };
    OuterProductAccumulate(outputErrorSignal, hiddenLayerOutput, Layer1Weights);
    VectorAccumulate(outputErrorSignal, networkInputs, layer1.biasesOffset);

    //RWMatrixRef<DATA_TYPE_FLOAT16, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> OuterMatrixDebug = { debugBuffer, 0, 0 };
    //OuterProductAccumulate(outputErrorSignal, hiddenLayerOutput, OuterMatrixDebug);       
    
    RWMatrixRef<DATA_TYPE_FLOAT16, NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> Layer0Weights = { networkInputs, layer0.weightsOffset, 0 };
    OuterProductAccumulate(hiddenErrorSignal, inputPixels, Layer0Weights);
    VectorAccumulate(hiddenErrorSignal, networkInputs, layer0.biasesOffset);

#endif
}