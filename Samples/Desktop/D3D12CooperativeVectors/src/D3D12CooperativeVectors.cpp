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

#include "stdafx.h"
#include "D3D12CooperativeVectors.h"
#include "Shared.h"
#include <DirectXMath.h>
#include <DirectXPackedVector.h>

#include "shaders.csh"
#include "Visualizer.csh"
#include <random>

#include "ClearShader.csh"
#include "InferenceCoopVec.csh"
#include "InferenceNoCoopVec.csh"
#include "TrainingCoopVec.csh"
#include "TrainingNoCoopVec.csh"

static const UINT MAX_BATCH_SIZE = 1E6;

extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = u8"."; }

uint GetSizeOfLinearAlgebraDatatype(D3D12_LINEAR_ALGEBRA_DATATYPE dataType)
{
    switch (dataType)
    {
        case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT_E4M3:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT_E5M2:
            return 1;
        case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT16:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT16:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16: 
            return 2;
        case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT32:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8_T4_PACKED:
        case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8_T4_PACKED:
            return 4;
        default: throw std::invalid_argument("Unknown data type");
    }
}

size_t AlignToMultipleOfPowerOfTwo(size_t value, size_t pow2Alignment)
{
    return (value + pow2Alignment - 1) & ~(pow2Alignment - 1);
}

D3D12CooperativeVectors::D3D12CooperativeVectors(UINT width, UINT height, std::wstring name) :
    DXSample(width, height, name),
    m_frameIndex(0),
    m_frameNumber(0)
{
}

void D3D12CooperativeVectors::OnInit()
{
    LoadPipeline();
    LoadAssets();
}

// Load the rendering pipeline dependencies.
void D3D12CooperativeVectors::LoadPipeline()
{
    // Enable experimental features if available.
    UUID Features[] = { D3D12ExperimentalShaderModels, D3D12CooperativeVectorExperiment };
    ThrowIfFailed(D3D12EnableExperimentalFeatures(_countof(Features), Features, nullptr, nullptr));

    UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
    // Enable the debug layer (requires the Graphics Tools "optional feature").
    // NOTE: Enabling the debug layer after device creation will invalidate the active device.
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();

            // Enable additional debug layers.
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;

            Microsoft::WRL::ComPtr<ID3D12Debug1> debugInterface1;
            if (SUCCEEDED((debugController->QueryInterface(IID_PPV_ARGS(&debugInterface1)))))
            {
                debugInterface1->SetEnableGPUBasedValidation(true);
            }
        }
    }
#endif

    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

    if (m_useWarpDevice)
    {
        ComPtr<IDXGIAdapter> warpAdapter;
        ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

        ThrowIfFailed(D3D12CreateDevice(
            warpAdapter.Get(),
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&m_device)
        ));
    }
    else
    {
        ComPtr<IDXGIAdapter1> hardwareAdapter;
        GetHardwareAdapter(factory.Get(), &hardwareAdapter);

        ThrowIfFailed(D3D12CreateDevice(
            hardwareAdapter.Get(),
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&m_device)
        ));
    }

    // Check for cooperative vector support.
    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL options = {};
    ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL, &options, sizeof(options)));

    if(options.CooperativeVectorTier >= D3D12_COOPERATIVE_VECTOR_TIER_1_0)
    {
        // PropCounts to be filled by driver implementation
        D3D12_FEATURE_DATA_COOPERATIVE_VECTOR CoopVecProperties = { 0, NULL, 0, NULL, 0, NULL };

        // CheckFeatureSupport returns the number of input combinations for intrinsics
        m_device->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR, &CoopVecProperties,
            sizeof(D3D12_FEATURE_DATA_COOPERATIVE_VECTOR));

        // Use MatrixVectorMulAddPropCount returned from the above

        // Use CheckFeatureSupport call to query only MatrixVectorMulAddProperties
        UINT MatrixVectorMulAddPropCount = CoopVecProperties.MatrixVectorMulAddPropCount;
        std::vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_MUL> properties(MatrixVectorMulAddPropCount);
        CoopVecProperties.pMatrixVectorMulAddProperties = properties.data();

        // CheckFeatureSupport returns the supported input combinations for the mul intrinsics
        m_device->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR, &CoopVecProperties,
            sizeof(D3D12_FEATURE_DATA_COOPERATIVE_VECTOR));

        int z = 0;
    }
    else
    {
        // Cooperative vectors are not supported.
        OutputDebugString(L"Cooperative vectors are not supported.\n");
        throw std::runtime_error("Cooperative vectors are not supported on this device.");
    }

    

    // Describe and create the command queue.
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

    // Describe and create the swap chain.
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = FrameCount;
    swapChainDesc.Width = m_width;
    swapChainDesc.Height = m_height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> swapChain;

    ThrowIfFailed(factory->CreateSwapChainForHwnd(
        m_commandQueue.Get(),        // Swap chain needs the queue so that it can force a flush on it.
        Win32Application::GetHwnd(),
        &swapChainDesc,
        nullptr,
        nullptr,
        &swapChain
    ));

    // This sample does not support fullscreen transitions.
    ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER));

    ThrowIfFailed(swapChain.As(&m_swapChain));
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    // Create descriptor heaps.
    {
        // Describe and create a render target view (RTV) descriptor heap.
        D3D12_DESCRIPTOR_HEAP_DESC uavHeapDesc = {};
        uavHeapDesc.NumDescriptors = 2;
        uavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        uavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(m_device->CreateDescriptorHeap(&uavHeapDesc, IID_PPV_ARGS(&m_uavHeap)));
    }

    for (UINT n = 0; n < FrameCount; n++)
    {
        ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
    }


    // Render Texture
    D3D12_RESOURCE_DESC textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        swapChainDesc.Format,
        m_width,
        m_height,
        1, // Array size
        1, // Mip levels
        1, // Sample count
        0, // Sample quality
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_visualizerTexture)));
    m_visualizerTexture->SetName(L"Visualizer Texture");

    // Create descriptor in the UAV heap for the visualizer texture.
    D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = m_uavHeap->GetCPUDescriptorHandleForHeapStart();
    m_device->CreateUnorderedAccessView(
        m_visualizerTexture.Get(),
        nullptr,
        nullptr,
        uavHandle);

    ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)));
}

// Load the sample assets.
void D3D12CooperativeVectors::LoadAssets()
{
    // Create the command list.
    ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), m_pipelineState.Get(), IID_PPV_ARGS(&m_commandList)));

    // Command lists are created in the recording state, but there is nothing
    // to record yet. The main loop expects it to be closed, so close it now.
    ThrowIfFailed(m_commandList->Close());

    // Create synchronization objects and wait until assets have been uploaded to the GPU.
    {
        ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
        m_fenceValue = 1;

        // Create an event handle to use for frame synchronization.
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (m_fenceEvent == nullptr)
        {
            ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
        }
    }

    ThrowIfFailed(m_device->CreateRootSignature(0, g_Visualizer, sizeof(g_Visualizer), IID_PPV_ARGS(&m_rootSignature)));

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.CS = { g_shaders, sizeof(g_shaders) };

    ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));
    m_pipelineState->SetName(L"Cooperative Vectors PSO");

    // Clear PSO
    psoDesc.CS = { g_ClearShader, sizeof(g_ClearShader) };
    ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_clearPSO)));

    // Visualizer PSO
    psoDesc.CS = { g_Visualizer, sizeof(g_Visualizer) };
    ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_visualizerPSO)));
    m_visualizerPSO->SetName(L"Visualizer PSO");

    // Training PSOs
    psoDesc.CS = { g_TrainingCoopVec, sizeof(g_TrainingCoopVec) };
    ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_trainingCoopVecPSO)));
    m_trainingCoopVecPSO->SetName(L"Training Cooperative Vector PSO");

    psoDesc.CS = { g_TrainingNoCoopVec, sizeof(g_TrainingNoCoopVec) };
    ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_trainingNoCoopVecPSO)));
    m_trainingNoCoopVecPSO->SetName(L"Training No Cooperative Vector PSO");

    // Inference PSOs
    psoDesc.CS = { g_InferenceCoopVec, sizeof(g_InferenceCoopVec) };
    ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_inferenceCoopVecPSO)));
    m_inferenceCoopVecPSO->SetName(L"Inference Cooperative Vector PSO");

    psoDesc.CS = { g_InferenceNoCoopVec, sizeof(g_InferenceNoCoopVec) };
    ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_inferenceNoCoopVecPSO)));
    m_inferenceNoCoopVecPSO->SetName(L"Inference No Cooperative Vector PSO");

    ComPtr<ID3D12DevicePreview> devicePreview;
    ThrowIfFailed(m_device.As(&devicePreview));

    uint layerSizes[3] = { NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS };
    
    size_t networkInputsSize_Upload = 0;
    size_t networkInputsSize_Default = 0;
    size_t trainingOutputsSize = 0;

    for(int i = 0; i < ARRAYSIZE(layerSizes) - 1; i++)
    {
        // For each layer, we need to create a conversion destination info.
        

        m_destInfos[i].DestLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL;
        m_destInfos[i].NumRows = layerSizes[i+1];
        m_destInfos[i].NumColumns = layerSizes[i];
        m_destInfos[i].DestStride = 0;// layerSizes[i + 1] * sizeof(DirectX::PackedVector::HALF);
        m_destInfos[i].DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;

        m_srcInfos[i].SrcDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        m_srcInfos[i].SrcLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
        m_srcInfos[i].SrcSize = m_destInfos[i].NumRows * m_destInfos[i].NumColumns * sizeof(DirectX::PackedVector::HALF);
        m_srcInfos[i].SrcStride = m_destInfos[i].NumColumns * sizeof(DirectX::PackedVector::HALF);

        devicePreview->GetLinearAlgebraMatrixConversionDestinationInfo(&m_destInfos[i]);

        // Default resource offsets
        {
            networkInputsSize_Default = AlignToMultipleOfPowerOfTwo(networkInputsSize_Default, 128);    // TODO. Ask D3D team for this as a constant
            m_layerOffsets[i].weightsOffset = (uint)networkInputsSize_Default;

            networkInputsSize_Default += m_destInfos[i].DestSize; // Weights matrix for Layer i

            networkInputsSize_Default = AlignToMultipleOfPowerOfTwo(networkInputsSize_Default, 64);     // TODO. Ask D3D team for this as a constant
            m_layerOffsets[i].biasesOffset = (uint)networkInputsSize_Default;

            networkInputsSize_Default += m_destInfos[i].NumRows * GetSizeOfLinearAlgebraDatatype(m_destInfos[i].DestDataType); // Biases vector for Layer i
        }

        // Upload offset and size calculation
        {
            networkInputsSize_Upload = AlignToMultipleOfPowerOfTwo(networkInputsSize_Upload, 4);    // Just in case Copy doesn't work with 2-byte alignment
            networkInputsSize_Upload += m_srcInfos[i].SrcSize;

            networkInputsSize_Upload = AlignToMultipleOfPowerOfTwo(networkInputsSize_Upload, 4);    // Just in case Copy doesn't work with 2-byte alignment
            networkInputsSize_Upload += (m_destInfos[i].NumRows * sizeof(DirectX::PackedVector::HALF));
        }

        // Forward pass size calculation
        {
            trainingOutputsSize += layerSizes[i+1] * GetSizeOfLinearAlgebraDatatype(m_destInfos[i].DestDataType); // Post-activation function values for Layer i
        }
    }

    trainingOutputsSize *= MAX_BATCH_SIZE;

    // Create a 1MB buffer in default heap to be used as a cooperative vector.
    const D3D12_HEAP_PROPERTIES defaultProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_HEAP_PROPERTIES uploadProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    D3D12_RESOURCE_DESC networkResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(networkInputsSize_Default, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    D3D12_RESOURCE_DESC networkUploadResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(networkInputsSize_Upload);
    D3D12_RESOURCE_DESC forwardPassResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(trainingOutputsSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    D3D12_RESOURCE_DESC debugResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(1 * 1024 * 1024, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    //D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(1 * 1024 * 1024); // Buffer for the cooperative vector.

    {
        // Weights and biases in an upload buffer, row major. Pre-copy.
        ThrowIfFailed(m_device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &networkUploadResourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_networkWeightsAndBiasesUpload)));
        m_networkWeightsAndBiasesUpload->SetName(L"Network Upload");
    }

    {
        ThrowIfFailed(m_device->CreateCommittedResource(
            &defaultProps,
            D3D12_HEAP_FLAG_NONE,
            &networkResourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_networkWeightsAndBiases)));
        m_networkWeightsAndBiases->SetName(L"Network");
    }

    {
        ThrowIfFailed(m_device->CreateCommittedResource(
            &defaultProps,
            D3D12_HEAP_FLAG_NONE,
            &forwardPassResourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_forwardPassOutput)));
        m_forwardPassOutput->SetName(L"Forward Output");
    }

    {
        ThrowIfFailed(m_device->CreateCommittedResource(
            &defaultProps,
            D3D12_HEAP_FLAG_NONE,
            &debugResourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_debugBuffer)));
        m_debugBuffer->SetName(L"Debug Buffer");
    }

    {
        D3D12_RESOURCE_DESC cbDesc = CD3DX12_RESOURCE_DESC::Buffer(65536);
        ThrowIfFailed(m_device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &cbDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_dynamicData)));
        m_dynamicData->SetName(L"Dynamic Constant Buffer");
    }

    {
        std::mt19937 rng;
        std::normal_distribution<float> dist(0.0f, 1.0f);

        void* pData;
        ThrowIfFailed(m_networkWeightsAndBiasesUpload->Map(0, nullptr, &pData));

        // Fill the whole input buffer with a simple pattern.
        UINT num2ByteValues = (UINT)m_networkWeightsAndBiasesUpload->GetDesc().Width / 2;
        USHORT* pShortData = static_cast<USHORT*>(pData);

        for (UINT i = 0; i < num2ByteValues; ++i)
        {
            float weight = INFINITY;
            pShortData[i] = DirectX::PackedVector::XMConvertFloatToHalf(weight);
        }

        char str[256];

        for(UINT l = 0; l < ARRAYSIZE(layerSizes) - 1; l++)
        {
            
            UINT numRows = m_destInfos[l].NumRows;
            UINT numCols = m_destInfos[l].NumColumns;
            UINT numWeights = numRows * numCols;
            UINT numBiases = numRows;

            for (UINT i = 0; i < numWeights; ++i)
            {
                float weight = dist(rng);
                pShortData[i] = DirectX::PackedVector::XMConvertFloatToHalf(weight);
            }

            for( UINT i = 0; i < numBiases; ++i)
            {
                float bias = dist(rng);
                pShortData[numWeights + i] = DirectX::PackedVector::XMConvertFloatToHalf(bias);
            }

            if (false)
            {
                sprintf_s(str, "Layer %d weights:\n", l);
                OutputDebugStringA(str);

                for (UINT i = 0; i < numWeights; ++i)
                {
                    float fp16Weight = DirectX::PackedVector::XMConvertHalfToFloat(pShortData[i]);
                    sprintf_s(str, "%f,", fp16Weight);
                    OutputDebugStringA(str);
                    if (i % numCols == (numCols - 1))
                        OutputDebugStringA("\n");
                }

                sprintf_s(str, "Layer %d biases:\n", l);
                OutputDebugStringA(str);

                for (UINT i = 0; i < numBiases; ++i)
                {
                    float fp16Bias = DirectX::PackedVector::XMConvertHalfToFloat(pShortData[numWeights + i]);
                    sprintf_s(str, "%f,", fp16Bias);
                    OutputDebugStringA(str);
                }
                OutputDebugStringA("\n");
            }

            pShortData += (numWeights + numBiases);
        }
    }

    // MNIST Data
    m_trainingSet.Load(m_device.Get(), "train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    m_testSet.Load(m_device.Get(), "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
}

// Update frame-based values.
void D3D12CooperativeVectors::OnUpdate()
{
    m_frameNumber++;
}

// Render the scene.
void D3D12CooperativeVectors::OnRender()
{
    // Record all the commands we need to render the scene into the command list.
    PopulateCommandList();

    // Execute the command list.
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present the frame.
    ThrowIfFailed(m_swapChain->Present(1, 0));

    WaitForPreviousFrame();
}

void D3D12CooperativeVectors::OnDestroy()
{
    // Ensure that the GPU is no longer referencing resources that are about to be
    // cleaned up by the destructor.
    WaitForPreviousFrame();

    CloseHandle(m_fenceEvent);
}

void D3D12CooperativeVectors::PopulateCommandList()
{
    static bool firstFrame = true;

    ComPtr<ID3D12DevicePreview> devicePreview;
    ThrowIfFailed(m_device.As(&devicePreview));
    
    ComPtr<ID3D12GraphicsCommandListPreview> commandListPreview;
    ThrowIfFailed(m_commandList.As(&commandListPreview));

    

    // Fill in the dynamic constant buffer
    size_t constantBufferSize = m_dynamicData->GetDesc().Width;
    void* pData;
    ThrowIfFailed(m_dynamicData->Map(0, nullptr, &pData));

    // Offset to the first or second half depending on the frame index
    UINT offset = m_frameIndex * (UINT)(constantBufferSize / 2);
    char* pCB = (char*)pData + offset;

    memcpy(pCB, &m_layerOffsets, sizeof(m_layerOffsets));

    m_networkOffsetsGPUVA = m_dynamicData->GetGPUVirtualAddress() + offset;

    // Command list allocators can only be reset when the associated 
    // command lists have finished execution on the GPU; apps should use 
    // fences to determine GPU execution progress.
    ThrowIfFailed(m_commandAllocator->Reset());

    // However, when ExecuteCommandList() is called on a particular command 
    // list, that command list can then be reset at any time and must be before 
    // re-recording.
    ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), m_pipelineState.Get()));

    if (firstFrame)
    {
        uint numLayers = ARRAYSIZE(m_destInfos);

        D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO convInfos[NUM_LAYERS - 1] = {};

        D3D12_GPU_VIRTUAL_ADDRESS uploadVA = m_networkWeightsAndBiasesUpload->GetGPUVirtualAddress();
        D3D12_GPU_VIRTUAL_ADDRESS networkVA = m_networkWeightsAndBiases->GetGPUVirtualAddress();

        size_t uploadOffset = 0;
        size_t networkOffset = 0;

        for (uint i = 0; i < numLayers; i++)
        {
            uploadOffset = AlignToMultipleOfPowerOfTwo(uploadOffset, 4);        // Upload alignment
            networkOffset = AlignToMultipleOfPowerOfTwo(networkOffset, 128);    // Weights alignment

            convInfos[i].DataDesc.SrcVA = uploadVA + uploadOffset;
            convInfos[i].DataDesc.DestVA = networkVA + networkOffset;

            convInfos[i].SrcInfo = m_srcInfos[i];
            convInfos[i].DestInfo = m_destInfos[i];

            uploadOffset += convInfos[i].SrcInfo.SrcSize;
            networkOffset += convInfos[i].DestInfo.DestSize;

            uploadOffset = AlignToMultipleOfPowerOfTwo(uploadOffset, 4);    // Upload alignment
            networkOffset = AlignToMultipleOfPowerOfTwo(networkOffset, 64); // Biases alignment

            // Copy biases
            UINT biasesSize = m_destInfos[i].NumRows * GetSizeOfLinearAlgebraDatatype(m_destInfos[i].DestDataType);

            commandListPreview->CopyBufferRegion(
                m_networkWeightsAndBiases.Get(), 
                networkOffset, 
                m_networkWeightsAndBiasesUpload.Get(), 
                uploadOffset, 
                biasesSize);

            uploadOffset += biasesSize;
            networkOffset += biasesSize;
        }

        D3D12_RESOURCE_BARRIER copyDestToUAVBarriers[1] = {};
        copyDestToUAVBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(m_networkWeightsAndBiases.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        m_commandList->ResourceBarrier(ARRAYSIZE(copyDestToUAVBarriers), copyDestToUAVBarriers);

        // Crashes PIX if it's inside a PIX capture
        commandListPreview->ConvertLinearAlgebraMatrix(convInfos, ARRAYSIZE(convInfos));

        D3D12_RESOURCE_BARRIER uavToNonPixelSRVBarriers[1] = {};
        uavToNonPixelSRVBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(m_networkWeightsAndBiases.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        m_commandList->ResourceBarrier(ARRAYSIZE(uavToNonPixelSRVBarriers), uavToNonPixelSRVBarriers);
    }
    

    if (firstFrame)
    {
        

        m_trainingSet.Upload(m_commandList.Get());
        m_testSet.Upload(m_commandList.Get());

        // Copy network 0's weight and biases from the upload buffers to the default ones
        //m_commandList->CopyResource(m_networkStorage[0].hiddenLayerWeightsBuffer.Get(), m_networkStorage[0].hiddenLayerWeightsBufferUpload.Get());
        //m_commandList->CopyResource(m_networkStorage[0].hiddenLayerBiasesBuffer.Get(), m_networkStorage[0].hiddenLayerBiasesBufferUpload.Get());
        //m_commandList->CopyResource(m_networkStorage[0].outputLayerWeightsBuffer.Get(), m_networkStorage[0].outputLayerWeightsBufferUpload.Get());
        //m_commandList->CopyResource(m_networkStorage[0].outputLayerBiasesBuffer.Get(), m_networkStorage[0].outputLayerBiasesBufferUpload.Get());
    }

    m_commandList->SetDescriptorHeaps(1, m_uavHeap.GetAddressOf());
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetComputeRootShaderResourceView(BindingSlots::NETWORK_OFFSETS_SRV, m_dynamicData->GetGPUVirtualAddress() + offset);

    // Zero out the m_forwardOutputs buffer
    {
        uint numFP16ValuesToClear = (UINT)(m_forwardPassOutput->GetDesc().Width / sizeof(DirectX::PackedVector::HALF));

        m_commandList->SetPipelineState(m_clearPSO.Get());
        m_commandList->SetComputeRoot32BitConstant(0, numFP16ValuesToClear, 0);
        m_commandList->SetComputeRootUnorderedAccessView(BindingSlots::NETWORK_WEIGHTS_AND_BIASES_UAV, m_forwardPassOutput->GetGPUVirtualAddress());
        
        uint numThreadGroupsNeeded = (numFP16ValuesToClear + 31) / 32;
        m_commandList->Dispatch(numThreadGroupsNeeded, 1, 1);
    }

    // Zero out the m_cooperativeVectorBufferDebug buffer
    {
        uint numFP16ValuesToClear = (UINT)(m_debugBuffer->GetDesc().Width / sizeof(DirectX::PackedVector::HALF));

        m_commandList->SetPipelineState(m_clearPSO.Get());
        m_commandList->SetComputeRoot32BitConstant(0, numFP16ValuesToClear, 0);
        m_commandList->SetComputeRootUnorderedAccessView(BindingSlots::NETWORK_WEIGHTS_AND_BIASES_UAV, m_debugBuffer->GetGPUVirtualAddress());

        uint numThreadGroupsNeeded = (numFP16ValuesToClear + 31) / 32;
        m_commandList->Dispatch(numThreadGroupsNeeded, 1, 1);
    }

    firstFrame = false;

    D3D12_RESOURCE_BARRIER preworkBarriers[2] = {};
    preworkBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
    //preworkBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_networkWeightsAndBiases.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    preworkBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_visualizerTexture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    //preworkBarriers[3] = CD3DX12_RESOURCE_BARRIER::UAV(m_cooperativeVectorBufferOutput.Get());

    m_commandList->ResourceBarrier(ARRAYSIZE(preworkBarriers), preworkBarriers);

    

    if (true)
    {
        static bool useCoopVec = true;

        if (useCoopVec)
            m_commandList->SetPipelineState(m_trainingCoopVecPSO.Get());
        else
            m_commandList->SetPipelineState(m_trainingNoCoopVecPSO.Get());

        UINT forwardPassStride = (NUM_HIDDEN_NEURONS + NUM_OUTPUT_NEURONS) * sizeof(DirectX::PackedVector::HALF);
        m_commandList->SetComputeRoot32BitConstant(BindingSlots::ROOT_CONSTANTS, forwardPassStride, 0);

        m_commandList->SetComputeRootShaderResourceView(BindingSlots::IMAGE_BUFFER, m_testSet.GetImageDataVA());
        m_commandList->SetComputeRootShaderResourceView(BindingSlots::LABEL_BUFFER, m_testSet.GetLabelsVA());

        m_commandList->SetComputeRootShaderResourceView(BindingSlots::NETWORK_WEIGHTS_AND_BIASES_SRV, m_networkWeightsAndBiases->GetGPUVirtualAddress());
        m_commandList->SetComputeRootShaderResourceView(BindingSlots::NETWORK_OFFSETS_SRV, m_dynamicData->GetGPUVirtualAddress());
        m_commandList->SetComputeRootUnorderedAccessView(BindingSlots::FORWARD_OUTPUTS_UAV, m_forwardPassOutput->GetGPUVirtualAddress());

        //m_commandList->Dispatch(m_testSet.GetNumImages(), 1, 1);
        m_commandList->Dispatch(65535, 1, 1);
    }

    D3D12_RESOURCE_BARRIER postCoopVecBarriers[1] = {};
    postCoopVecBarriers[0] = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    
    m_commandList->ResourceBarrier(ARRAYSIZE(postCoopVecBarriers), postCoopVecBarriers);

    /*
    if (false)
    {
        // Copy the OUTER_PRODUCT_OPTIMAL output to a ROW_MAJOR debug output

        D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO debugConvInfo = {};
        debugConvInfo.DataDesc.SrcVA = m_cooperativeVectorBufferOutput->GetGPUVirtualAddress();
        debugConvInfo.SrcInfo.SrcDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        debugConvInfo.SrcInfo.SrcLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL;
        debugConvInfo.SrcInfo.SrcSize = m_destInfos[1].DestSize;
        debugConvInfo.SrcInfo.SrcStride = 0;

        debugConvInfo.DataDesc.DestVA = m_cooperativeVectorBufferDebug->GetGPUVirtualAddress();
        debugConvInfo.DestInfo.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        debugConvInfo.DestInfo.DestLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
        debugConvInfo.DestInfo.NumRows = NUM_OUTPUT_NEURONS;
        debugConvInfo.DestInfo.NumColumns = NUM_HIDDEN_NEURONS;
        debugConvInfo.DestInfo.DestStride = debugConvInfo.DestInfo.NumColumns * sizeof(DirectX::PackedVector::HALF);

        devicePreview->GetLinearAlgebraMatrixConversionDestinationInfo(&debugConvInfo.DestInfo);

        commandListPreview->ConvertLinearAlgebraMatrix(&debugConvInfo, 1);
    }*/

    // Visualizer pass
    if(true)
    {
        m_commandList->SetPipelineState(m_visualizerPSO.Get());
        m_commandList->SetComputeRootShaderResourceView(BindingSlots::IMAGE_BUFFER, m_testSet.GetImageDataVA()); // This!
        m_commandList->SetComputeRootShaderResourceView(BindingSlots::LABEL_BUFFER, m_testSet.GetLabelsVA());

        m_commandList->SetComputeRootShaderResourceView(BindingSlots::NETWORK_WEIGHTS_AND_BIASES_SRV, m_networkWeightsAndBiases->GetGPUVirtualAddress());

        m_commandList->Dispatch(1, 1, 1);
    }

    // TODO. Enhanced Barriers?
    
    D3D12_RESOURCE_BARRIER postVisualizerBarriers[1] = {};
    postVisualizerBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(m_visualizerTexture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    //postVisualizerBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_cooperativeVectorBufferInputMulOptimalWeights.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Indicate that the back buffer will now be used to present.
    m_commandList->ResourceBarrier(ARRAYSIZE(postVisualizerBarriers), postVisualizerBarriers);

    // Copy the visualizer texture to the back buffer.
    m_commandList->CopyResource(m_renderTargets[m_frameIndex].Get(), m_visualizerTexture.Get());

    // Transition the back buffer to be ready for presentation.
    D3D12_RESOURCE_BARRIER postRenderBarriers[1] = {};
    postRenderBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);

    m_commandList->ResourceBarrier(ARRAYSIZE(postRenderBarriers), postRenderBarriers);

    ThrowIfFailed(m_commandList->Close());
}

void D3D12CooperativeVectors::WaitForPreviousFrame()
{
    // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
    // This is code implemented as such for simplicity. The D3D12HelloFrameBuffering
    // sample illustrates how to use fences for efficient resource usage and to
    // maximize GPU utilization.

    // Signal and increment the fence value.
    const UINT64 fence = m_fenceValue;
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fence));
    m_fenceValue++;

    // Wait until the previous frame is finished.
    if (m_fence->GetCompletedValue() < fence)
    {
        ThrowIfFailed(m_fence->SetEventOnCompletion(fence, m_fenceEvent));
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
}
