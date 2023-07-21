#pragma once
#define DLL_EXPORT __declspec(dllexport)
#include "detector.h"
#include <ctime>

//创建实例
extern "C" DLL_EXPORT UnetDectector * CreateDetector_CPU();
//初始化设备
extern "C" DLL_EXPORT void InitializeDetector_CPU(UnetDectector * m_Unet,
	const char* device_char, const char* xml_path_char);
//推理
extern "C" DLL_EXPORT void PredictDetector_CPU(UnetDectector * m_Unet,
	unsigned char* image_batch, unsigned char* prediction_batch, 
	int width, int height, int smallestMax , int batch_size );
//析构实例
extern "C" DLL_EXPORT void DisposeDetector_CPU(UnetDectector * m_Unet);