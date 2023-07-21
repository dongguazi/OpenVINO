#pragma once
#define DLL_EXPORT __declspec(dllexport)
#include "detector.h"
#include <ctime>

//����ʵ��
extern "C" DLL_EXPORT UnetDectector * CreateDetector_CPU();
//��ʼ���豸
extern "C" DLL_EXPORT void InitializeDetector_CPU(UnetDectector * m_Unet,
	const char* device_char, const char* xml_path_char);
//����
extern "C" DLL_EXPORT void PredictDetector_CPU(UnetDectector * m_Unet,
	unsigned char* image_batch, unsigned char* prediction_batch, 
	int width, int height, int smallestMax , int batch_size );
//����ʵ��
extern "C" DLL_EXPORT void DisposeDetector_CPU(UnetDectector * m_Unet);