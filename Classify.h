#pragma once
#define DLL_EXPORT __declspec(dllexport)
#include "detector.h"
#include <ctime>

extern "C" DLL_EXPORT ClassifyClass * CreateClassify_CPU();
//
extern "C" DLL_EXPORT void InitializeClassify_CPU(ClassifyClass * m_Classify,
	const char* device_char, const char* onnx_path_char);
//
extern "C" DLL_EXPORT void PredictClassify_CPU(ClassifyClass * m_Classify,
	unsigned char* image_batch, float* prediction_batch, 
	int width, int height, int training_size, int batch_size );
//
extern "C" DLL_EXPORT void DisposeClassify_CPU(ClassifyClass * m_Classify);