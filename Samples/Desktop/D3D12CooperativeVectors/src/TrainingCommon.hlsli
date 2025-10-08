#include "Header.hlsli"

#if defined(USE_COOPERATIVE_VECTORS)
#include <dx/linalg.h>
using namespace dx::linalg;
#endif
                                             
static const uint NUM_INPUTS_TO_LOAD = NUM_INPUT_NEURONS;

cbuffer Params
{
    uint forwardPassStride; // in bytes
    uint numImages;
    uint firstImage;
};   

struct DebugOut
{
    float16_t preActValue;
    float16_t postActValue;
    float16_t loss;
};                  

RWByteAddressBuffer forwardPassOutput : register(u1);
RWByteAddressBuffer accumulatedGradientsOutput : register(u3);

ByteAddressBuffer networkInputs : register(t2);

[RootSignature(RootSig)]
[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{  
    if(DTid.x >= numImages)
        return;

    uint imageIndex = DTid.x + firstImage;

#if defined(USE_COOPERATIVE_VECTORS)
    vector<float16_t, NUM_INPUTS_TO_LOAD> inputPixels = GetPixels<NUM_INPUTS_TO_LOAD>(imageIndex, 0);
    
    NetworkOffsets layer0 = g_networkOffsets[0];
    NetworkOffsets layer1 = g_networkOffsets[1];

    MatrixRef<DATA_TYPE_FLOAT16, NUM_HIDDEN_NEURONS, NUM_INPUTS_TO_LOAD, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> HiddenLayerWeights = { networkInputs, layer0.weightsOffset, 0 };
    VectorRef<DATA_TYPE_FLOAT16> HiddenLayerBiases = {networkInputs, layer0.biasesOffset };

    vector<float16_t, NUM_HIDDEN_NEURONS> hiddenLayerOutput = MulAdd<float16_t>(HiddenLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(inputPixels), HiddenLayerBiases);
    ActivationFunction(hiddenLayerOutput);
        
    MatrixRef<DATA_TYPE_FLOAT16, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> OutputLayerWeights = { networkInputs, layer1.weightsOffset, 0 };
    MatrixRef<DATA_TYPE_FLOAT16, NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL, true> OutputLayerWeightsTransposed = { networkInputs, layer1.weightsOffset, 0 };
    VectorRef<DATA_TYPE_FLOAT16> OutputLayerBiases = {networkInputs, layer1.biasesOffset };
    
    vector<float16_t, NUM_OUTPUT_NEURONS> outputLayerOutput = MulAdd<float16_t>(OutputLayerWeights, MakeInterpretedVector<DATA_TYPE_FLOAT16>(hiddenLayerOutput), OutputLayerBiases);
    
    ActivationFunction(outputLayerOutput); 
    
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

    vector<float16_t, NUM_OUTPUT_NEURONS> output_error_signal;

    for(uint16_t outputNeuronIndex = 0; outputNeuronIndex < NUM_OUTPUT_NEURONS; outputNeuronIndex++)
    {                                
        float16_t neuronOutput = outputLayerOutput[outputNeuronIndex];
        float16_t desiredOutput = (label == outputNeuronIndex) ? 1.0h : 0.0h;
        
        float16_t pd_Error_Output = (neuronOutput - desiredOutput);         
        //debugOutput[outputNeuronIndex].loss = pd_Error_Output;
		float16_t pd_Output_Sum = ActivationFunctionDerivative(neuronOutput);
          
        output_error_signal[outputNeuronIndex] = pd_Error_Output * -pd_Output_Sum;
    }                  
    		
	vector<float16_t, NUM_HIDDEN_NEURONS> hidden_error_signal = Mul<float16_t>(OutputLayerWeightsTransposed, MakeInterpretedVector<DATA_TYPE_FLOAT16>(output_error_signal));
	hidden_error_signal *= ActivationFunctionDerivative(hiddenLayerOutput);
		
    //RWMatrixRef<DATA_TYPE_FLOAT32, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL> Layer1Weights = { accumulatedGradientsOutput, layer1.accumulatedWeightsOffset, 0 };
    //OuterProductAccumulate(output_error_signal, hiddenLayerOutput, Layer1Weights);
    //VectorAccumulate(output_error_signal, accumulatedGradientsOutput, layer1.accumulatedBiasesOffset);

    // To update the output layer gradients we need:
    // Output Error Signal x Hidden Layer Post Activation
    // To update the hidden layer gradients we need:
    // Input x Hidden Error Signal
    // So we need to write output hidden layer post activation, hidden error signal, output error signal

    uint outputBase = (imageIndex * forwardPassStride);
    forwardPassOutput.Store(outputBase, hiddenLayerOutput);

    outputBase += sizeof(hiddenLayerOutput);
    forwardPassOutput.Store(outputBase, hidden_error_signal);

    outputBase += sizeof(hidden_error_signal);
    forwardPassOutput.Store(outputBase, output_error_signal);
#endif
}