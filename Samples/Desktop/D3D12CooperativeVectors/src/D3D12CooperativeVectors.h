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

#pragma once

#include "DXSample.h"
#include "Dataset.h"
#include "Shared.h"

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

class D3D12CooperativeVectors : public DXSample
{
public:
    D3D12CooperativeVectors(UINT width, UINT height, std::wstring name);

    virtual void OnInit();
    virtual void OnUpdate();
    virtual void OnRender();
    virtual void OnDestroy();

private:
    static const UINT FrameCount = 2;

    // Pipeline objects.
    ComPtr<IDXGISwapChain3> m_swapChain;
    ComPtr<ID3D12Device2> m_device;
    ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12DescriptorHeap> m_uavHeap;
    
    ComPtr<ID3D12GraphicsCommandList1> m_commandList;

    ComPtr<ID3D12PipelineState> m_clearPSO;
    ComPtr<ID3D12PipelineState> m_pipelineState, m_visualizerPSO;
    ComPtr<ID3D12PipelineState> m_trainingCoopVecPSO, m_trainingNoCoopVecPSO;
    ComPtr<ID3D12PipelineState> m_inferenceCoopVecPSO, m_inferenceNoCoopVecPSO;

    // App resources.
    ComPtr<ID3D12Resource> m_visualizerTexture;
    ComPtr<ID3D12Resource> m_networkWeightsAndBiases, m_networkWeightsAndBiasesUpload;
    ComPtr<ID3D12Resource> m_forwardPassOutput;
    //ComPtr<ID3D12Resource> m_cooperativeVectorBufferInputMulOptimalWeights;
    //ComPtr<ID3D12Resource> m_cooperativeVectorBufferInputBiases;
    //ComPtr<ID3D12Resource> m_cooperativeVectorBufferOutput;
    ComPtr<ID3D12Resource> m_debugBuffer;

    ComPtr<ID3D12Resource> m_dynamicData;
    D3D12_GPU_VIRTUAL_ADDRESS m_networkOffsetsGPUVA = 0;

    static const UINT NUM_LAYERS = 3;
    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_SRC_INFO m_srcInfos[NUM_LAYERS - 1] = {};
    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO m_destInfos[NUM_LAYERS-1] = {};
    NetworkOffsets m_layerOffsets[NUM_LAYERS-1];
    
    // Dataset resources.
    Dataset m_trainingSet, m_testSet;

    // Synchronization objects.
    UINT m_frameIndex;
    UINT m_frameNumber;
    HANDLE m_fenceEvent;
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue;

    void LoadPipeline();
    void LoadAssets();
    void PopulateCommandList();
    void WaitForPreviousFrame();
};
