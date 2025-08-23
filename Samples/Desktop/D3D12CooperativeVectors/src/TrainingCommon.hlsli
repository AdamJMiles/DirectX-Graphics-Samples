#include "Header.hlsli"

#if defined(USE_COOPERATIVE_VECTORS)
#include <dx/linalg.h>
using namespace dx::linalg;
#endif

static const uint NUM_INPUTS_TO_LOAD = 4;

cbuffer Params
{
    uint weightsOffset;
    uint biasesOffset;
};

RWByteAddressBuffer output : register(u0);

ByteAddressBuffer inputWeights : register(t2);
ByteAddressBuffer inputBiases : register(t3);

[RootSignature(RootSig)]
[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{  
    //if(groupID.x != 0)
    //    return;

    uint imageIndex = 0;
    uint pixelInputOffset = 0;//groupID.x * NUM_INPUTS_TO_LOAD;

    uint hiddenLayerWeightsOffset = 0;
    uint hiddenLayerBiasesOffset = 0;
        
    uint outputLayerMatrixOffset = 50176;
    uint outputLayerBiasesOffset = NUM_HIDDEN_NEURONS * sizeof(float16_t);

#if defined(USE_COOPERATIVE_VECTORS)
    vector<float16_t, NUM_INPUTS_TO_LOAD> inputPixel = GetPixels<NUM_INPUTS_TO_LOAD>(imageIndex, pixelInputOffset);
    
    for(int i = 0; i < NUM_INPUTS_TO_LOAD; i++)
        inputPixel[i] = 1.0f;  
        
    //inputPixel[0] = 1.0f; 
    //inputPixel[1] = 1.0f;
    //inputPixel[2] = 0.0f;
    //inputPixel[3] = 0.0f;

    VectorRef<DATA_TYPE_FLOAT16> HiddenLayerBiases = {inputBiases, hiddenLayerBiasesOffset };
    MatrixRef<DATA_TYPE_FLOAT16, NUM_HIDDEN_NEURONS, NUM_INPUTS_TO_LOAD, MATRIX_LAYOUT_ROW_MAJOR> HiddenLayerWeights = { inputWeights, hiddenLayerWeightsOffset, 0 };

    vector<float16_t, NUM_HIDDEN_NEURONS> hiddenLayerOutput = Mul<float16_t>(HiddenLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(inputPixel));
    
    //ActivationFunction(hiddenLayerOutput);
    output.Store<vector<float16_t, NUM_HIDDEN_NEURONS> >(0, hiddenLayerOutput);
    /*
    VectorRef<DATA_TYPE_FLOAT16> OutputLayerBiases = {inputBiases, outputLayerBiasesOffset };
    MatrixRef<DATA_TYPE_FLOAT16, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_ROW_MAJOR> OutputLayerWeights = { inputWeights, outputLayerMatrixOffset, 0 };

    vector<float16_t, NUM_OUTPUT_NEURONS> outputLayerOutput = MulAdd<float16_t>(OutputLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(hiddenLayerOutput), OutputLayerBiases);

    ActivationFunction(outputLayerOutput);

    output.Store<vector<float16_t, NUM_OUTPUT_NEURONS> >(0, outputLayerOutput);

    // Now do back propagation
    uint16_t label = GetLabel(imageIndex);

    // Calculate the highest output neuron
    uint16_t maxIndex = 0;
    float16_t maxValue = outputLayerOutput[0];

    for(uint16_t i = 1; i < NUM_OUTPUT_NEURONS; i++)
    {
        if(outputLayerOutput[i] > maxValue)
        {
            maxValue = outputLayerOutput[i];
            maxIndex = i;
        }
    }

    static const float16_t LEARNING_RATE = 1.0h;
    float16_t pd_Activation_Output = maxValue * (1.0h - maxValue);

    vector<float16_t, NUM_OUTPUT_NEURONS> eo_os_lr;

    for(uint16_t i = 0; i < NUM_OUTPUT_NEURONS; i++)
    {
        float16_t expectedValueI = (i == label) ? 1.0h : 0.0h;
        float16_t pd_Error_Output = maxValue - expectedValueI;
        eo_os_lr[i] = pd_Error_Output * pd_Activation_Output * LEARNING_RATE;
    }

    RWMatrixRef<DATA_TYPE_FLOAT16, 32, 10, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> OuterMatrix = { output, 4096, 0 };
    OuterProductAccumulate(hiddenLayerOutput, eo_os_lr, OuterMatrix);

    VectorAccumulate(eo_os_lr, output, 8192);
    */
#endif
}