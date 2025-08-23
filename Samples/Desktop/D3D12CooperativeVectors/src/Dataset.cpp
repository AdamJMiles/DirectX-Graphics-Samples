#include "stdafx.h"
#include "Dataset.h"

Dataset::Dataset()
{
}

Dataset::~Dataset()
{
}

void FourByteEndianSwap(UINT* data, UINT count)
{
	for (UINT i = 0; i < count; i++)
	{
		data[i] = _byteswap_ulong(data[i]);
	}
}

void Dataset::Load(ID3D12Device* device, const char* imagesFilename, const char* labelsFilename)
{
	HANDLE fh = CreateFileA(imagesFilename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

	if (fh == INVALID_HANDLE_VALUE)
	{
		return;
	}

	DWORD fileSize = GetFileSize(fh, NULL);

	UINT header[4];
	DWORD bytesRead;
	ReadFile(fh, header, sizeof(header), &bytesRead, NULL);

	FourByteEndianSwap(header, ARRAYSIZE(header));
	assert(header[2] == 28 && header[3] == 28);

	m_numImages = header[1];
	m_imageWidth = header[2];
	m_imageHeight = header[3];

	UINT totalImageSize = m_numImages * m_imageWidth * m_imageHeight;
	void* imageData = malloc(totalImageSize);
	ReadFile(fh, imageData, totalImageSize, &bytesRead, NULL);
	assert(bytesRead == totalImageSize);

	CloseHandle(fh);


    // Print out the first image for debugging
    for (UINT y = 0; y < m_imageHeight; y++)
    {
        for (UINT x = 0; x < m_imageWidth; x++)
        {
            UINT8 pixel = ((UINT8*)imageData)[y * m_imageWidth + x];
            float f = (pixel / 255.0f);
            char str[256];
            sprintf_s(str, "%f\n", f);
            //OutputDebugStringA(str);
        }
        //OutputDebugStringA("\n");
    }


	fh = CreateFileA(labelsFilename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

	if (fh == INVALID_HANDLE_VALUE)
	{
		return;
	}

	fileSize = GetFileSize(fh, NULL);

	UINT labelsHeader[2];
	ReadFile(fh, labelsHeader, sizeof(labelsHeader), &bytesRead, NULL);
	FourByteEndianSwap(labelsHeader, ARRAYSIZE(labelsHeader));

	assert(labelsHeader[1] == m_numImages);

	void* labelData = (uint8_t*)malloc(m_numImages);
	ReadFile(fh, labelData, m_numImages, &bytesRead, NULL);
	assert(bytesRead == m_numImages);

	CloseHandle(fh);

	m_imageDataResUpload = AllocateResource(device, D3D12_HEAP_TYPE_UPLOAD, imageData, totalImageSize);
	m_labelsResUpload = AllocateResource(device, D3D12_HEAP_TYPE_UPLOAD, labelData, m_numImages);

	m_imageDataResDefault = AllocateResource(device, D3D12_HEAP_TYPE_DEFAULT, nullptr, totalImageSize);
	m_labelsResDefault = AllocateResource(device, D3D12_HEAP_TYPE_DEFAULT, nullptr, m_numImages);

	free(imageData);
	free(labelData);
}

Microsoft::WRL::ComPtr<ID3D12Resource> Dataset::AllocateResource(ID3D12Device* device, D3D12_HEAP_TYPE heapType, void* data, UINT size)
{
	Microsoft::WRL::ComPtr<ID3D12Resource> res;

	CD3DX12_HEAP_PROPERTIES heapProps(heapType);
	CD3DX12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(size);

	device->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&resDesc,
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&res)
	);

	if (data != nullptr)
	{
		void* mappedData;
		res->Map(0, nullptr, &mappedData);
		memcpy(mappedData, data, size);
		res->Unmap(0, nullptr);
	}

	return res;
}

void Dataset::Upload(ID3D12GraphicsCommandList* cmdList)
{
	cmdList->CopyResource(m_imageDataResDefault.Get(), m_imageDataResUpload.Get());
	cmdList->CopyResource(m_labelsResDefault.Get(), m_labelsResUpload.Get());

	D3D12_RESOURCE_BARRIER barriers[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(m_imageDataResDefault.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(m_labelsResDefault.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
	};

	cmdList->ResourceBarrier(2, barriers);
}