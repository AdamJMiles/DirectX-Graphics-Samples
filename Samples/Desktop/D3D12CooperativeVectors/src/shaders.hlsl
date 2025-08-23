//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "Header.hlsli"
#include <dx/linalg.h>

ByteAddressBuffer inputWeights : register(t0);
ByteAddressBuffer inputBiases : register(t1);

RWByteAddressBuffer output : register(u0);

using namespace dx::linalg;

#define WAVE_SIZE 32
#define INPUT_SIZE 32

[RootSignature(RootSig)]
[numthreads(WAVE_SIZE,1,1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    vector<float16_t, INPUT_SIZE> inputVector;
    for(int i = 0; i < INPUT_SIZE; i++)
        inputVector[i] = 1.0f;

    VectorRef<DATA_TYPE_FLOAT16> Biases = {inputBiases, 0 };
    MatrixRef<DATA_TYPE_FLOAT16, INPUT_SIZE, INPUT_SIZE, MATRIX_LAYOUT_MUL_OPTIMAL> MulMatrix = { inputWeights, 0, 0 };
    
    vector<float16_t, INPUT_SIZE> layerOutput = inputVector;
    
    [unroll]
    for(int i = 0; i < 1; i++)
    {
        layerOutput = MulAdd<float16_t>(MulMatrix, MakeInterpretedVector<DATA_TYPE_FLOAT16>(layerOutput), Biases);
    }
    
    uint address = DTid.x * INPUT_SIZE * 2;
    output.Store<vector<float16_t, INPUT_SIZE> >(address, layerOutput);
    //VectorAccumulate(layerOutput, output, 0);
}