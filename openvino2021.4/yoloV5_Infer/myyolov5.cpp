#include"myyolov5.h"

extern "C" Yolov5Dectector * CreateYolov5Parser_CPU()
{
	Yolov5Dectector* m_yolov5 = new Yolov5Dectector();
	return m_yolov5;
}

extern "C" void InitializeDetector_CPU(Yolov5Dectector* m_yolov5,
	const char* device_char, const char* xml_path_char)
{
	//类型转换
	string device(device_char);
	string xml_path(xml_path_char);
	m_yolov5->InitializeDetector(device, xml_path);
}

extern "C" void PredictDetector_CPU(Yolov5Dectector * m_yolov5, 
	unsigned char* image_batch, 
	float* x1_ptr, float* y1_ptr, float* x2_ptr, float* y2_ptr, 
	float* prob_ptr, int* class_ptr, int* num_boxes, int buffer_size, 
	int width, int height, int batch_size, float conf_thresh, float nms_thres)
{
	m_yolov5->process_frame(image_batch, x1_ptr, y1_ptr, x2_ptr, y2_ptr, prob_ptr, class_ptr, num_boxes, buffer_size, width, height, batch_size, conf_thresh, nms_thres);
}

extern "C"  void DisposeDetector_CPU(Yolov5Dectector * m_yolov5)
{
	delete m_yolov5;
}