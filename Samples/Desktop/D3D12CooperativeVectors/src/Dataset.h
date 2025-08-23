#pragma once
class Dataset
{
public:
	Dataset();
	~Dataset();

	void Load(ID3D12Device* device, const char* imagesFilename, const char* labelsFilename);

	void Upload(ID3D12GraphicsCommandList* cmdList);

	Microsoft::WRL::ComPtr<ID3D12Resource> AllocateResource(ID3D12Device* device, D3D12_HEAP_TYPE heapType, void* data = nullptr, UINT size = 0);

	D3D12_GPU_VIRTUAL_ADDRESS GetImageDataVA() const
	{
		return m_imageDataResDefault->GetGPUVirtualAddress();
	}

	D3D12_GPU_VIRTUAL_ADDRESS GetLabelsVA() const
	{
		return m_labelsResDefault->GetGPUVirtualAddress();
	}

	UINT GetNumPixels() const
	{
		return m_imageWidth * m_imageHeight;
	}

	UINT GetNumImages() const
	{
		return m_numImages;
	}

private:
	UINT m_imageWidth, m_imageHeight;
	UINT m_numImages;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_imageDataResUpload;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_labelsResUpload;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_imageDataResDefault;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_labelsResDefault;
};

