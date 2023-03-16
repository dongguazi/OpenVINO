#include"Unet.h"


extern "C"  UnetDectector * CreateDetector_CPU() 
{
	UnetDectector* m_Unet = new UnetDectector();
	return m_Unet;
}
//
extern "C"  void InitializeDetector_CPU(UnetDectector * m_Unet,
	const char* device_char, const char* xml_path_char)
{
	//ÀàÐÍ×ª»»
	string device(device_char);
	string xml_path(xml_path_char);
	m_Unet->InitializeDetector(device, xml_path);
}
//
extern "C"  void PredictDetector_CPU(UnetDectector * m_Unet,
	unsigned char* image_batch, unsigned char* prediction_batch,
	int width, int height, int smallestMax = 512, int batch_size = 1)
{
	m_Unet->process_frame(image_batch, prediction_batch,width,height, smallestMax, batch_size);
}
//
extern "C" void DisposeDetector_CPU(UnetDectector * m_Unet) {

	delete m_Unet;
}