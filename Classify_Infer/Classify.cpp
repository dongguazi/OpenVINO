#include"Classify.h"


extern "C"  ClassifyClass * CreateClassify_CPU()
{
	ClassifyClass* m_Classify = new ClassifyClass();
	return m_Classify;
}
//
extern "C"  void InitializeClassify_CPU(ClassifyClass * m_Classify,
	const char* device_char, const char* onnx_path_char)
{
	//ÀàÐÍ×ª»»
	string device(device_char);
	string onnx_path(onnx_path_char);
	m_Classify->InitializeDetector(device, onnx_path);
}
//
extern "C"  void PredictClassify_CPU(ClassifyClass * m_Classify,
	unsigned char* image_batch, float* prediction_batch,
	int width, int height, int training_size, int batch_size = 1)
{
	m_Classify->process_frame(image_batch, prediction_batch,width,height, training_size, batch_size);
}
//
extern "C" void DisposeClassify_CPU(ClassifyClass * m_Classify) {

	delete m_Classify;
}