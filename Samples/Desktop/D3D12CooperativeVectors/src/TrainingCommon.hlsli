#include "Header.hlsli"

#if defined(USE_COOPERATIVE_VECTORS)
#include <dx/linalg.h>
using namespace dx::linalg;
#endif
                                             
static const float16_t LEARNING_RATE = 1.0h;
static const uint NUM_INPUTS_TO_LOAD = 16;

cbuffer Params
{
    uint forwardPassStride; // in bytes
};

RWByteAddressBuffer forwardPassOutput : register(u1);

ByteAddressBuffer networkInputs : register(t2);
//ByteAddressBuffer inputBiases : register(t3);

[RootSignature(RootSig)]
[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{  
    uint imageIndex = 0;//groupID.x;
    uint batchIndex = groupID.x;

#if defined(USE_COOPERATIVE_VECTORS)
    vector<float16_t, NUM_INPUTS_TO_LOAD> inputPixel = GetPixels<NUM_INPUTS_TO_LOAD>(imageIndex, 0);
    
    NetworkOffsets layer0 = g_networkOffsets[0];
    NetworkOffsets layer1 = g_networkOffsets[1];

    MatrixRef<DATA_TYPE_FLOAT16, NUM_HIDDEN_NEURONS, NUM_INPUTS_TO_LOAD, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> HiddenLayerWeights = { networkInputs, layer0.weightsOffset, 0 };
    VectorRef<DATA_TYPE_FLOAT16> HiddenLayerBiases = {networkInputs, layer0.biasesOffset };

    vector<float16_t, NUM_HIDDEN_NEURONS> hiddenLayerOutput = MulAdd<float16_t>(HiddenLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(inputPixel), HiddenLayerBiases);
    ActivationFunction(hiddenLayerOutput);
        
   // MatrixRef<DATA_TYPE_FLOAT16, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> OutputLayerWeights = { networkInputs, layer1.weightsOffset, 0 };
   // VectorRef<DATA_TYPE_FLOAT16> OutputLayerBiases = {networkInputs, layer1.biasesOffset };

   // vector<float16_t, NUM_OUTPUT_NEURONS> outputLayerOutput = MulAdd<float16_t>(OutputLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(hiddenLayerOutput), OutputLayerBiases);
   // ActivationFunction(outputLayerOutput);

    // Write the output
    uint outputBase = (batchIndex * forwardPassStride);
    forwardPassOutput.Store<vector<float16_t, NUM_HIDDEN_NEURONS> >(outputBase, hiddenLayerOutput);
/*
    outputBase += sizeof(float16_t) * NUM_HIDDEN_NEURONS;
    forwardPassOutput.Store<vector<float16_t, NUM_OUTPUT_NEURONS> >(outputBase, outputLayerOutput);
          
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

    for(uint16_t outputNeuronIndex = 0; outputNeuronIndex < NUM_OUTPUT_NEURONS; outputNeuronIndex++)
    {                                
        float16_t neuronOutput = outputLayerOutput[outputNeuronIndex];
        float16_t desiredOutput = (label == outputNeuronIndex) ? 1.0h : 0.0h;
        
        float16_t pd_Error_Output = (neuronOutput - desiredOutput); // How far off were we?
        float16_t pd_Output_Sum = neuronOutput * (1 - neuronOutput); 
          
        eo_os_lr[outputNeuronIndex] = pd_Error_Output * pd_Output_Sum * LEARNING_RATE;
    } 
    
    //output.Store<vector<float16_t, NUM_OUTPUT_NEURONS> >(0, eo_os_lr);

    //RWMatrixRef<DATA_TYPE_FLOAT16, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> OuterMatrix = { output, 0, 0 };
    //OuterProductAccumulate(eo_os_lr, hiddenLayerOutput, OuterMatrix); 
                            
    uint addr = 0;
    for(int h = 0; h < 32; h++)
    {
        for(int o = 0; o < 10; o++)
        {
             float16_t v = hiddenLayerOutput[h] * eo_os_lr[o];
             //output.Store<float16_t>(addr, v);
             addr += 2;
        }
    }

    //VectorAccumulate(eo_os_lr, output, 0);
      */ 
#endif
}