#pragma once
#include<string>
#include<ie/inference_engine.hpp>
#include<samples/ocv_common.hpp>
#include<ngraph/ngraph.hpp>
#include<opencv2/opencv.hpp>
#include<time.h>

using namespace InferenceEngine;
using namespace std;
using namespace cv;
typedef unsigned char unit8_t;
class Yolov4Dectector
{
public:
   struct YoloV4Result
	{
		int class_num = -1;
		float prob = 0.f;
		float x1 = 0.f;
		float y1 = 0.f;
		float x2 = 0.f;
		float y2 = 0.f;
	};

	Yolov4Dectector();

	~Yolov4Dectector();

	vector<YoloV4Result> yolov4_result;
	/// <summary>
	/// ��ʼ��
	/// </summary>
	/// <param name="device"></param>
	/// <param name="xml_path"></param>
	/// <returns></returns>
	bool InitializeDetector(string device,string xml_path);

	bool uninit();

	/// <summary>
	/// ����
	/// </summary>
	/// <param name="image_batch"></param>
	/// <param name="x1_ptr"></param>
	/// <param name="y1_ptr"></param>
	/// <param name="x2_ptr"></param>
	/// <param name="y2_ptr"></param>
	/// <param name="prob_ptr"></param>
	/// <param name="class_ptr"></param>
	/// <param name="num_boxes"></param>
	/// <param name="buffer_size"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="batch_size"></param>
	/// <param name="conf_thresh"></param>
	/// <param name="nms_thres"></param>
	/// <returns></returns>
	void process_frame(unsigned char* image_batch,
		float* x1_ptr, float* y1_ptr, float* x2_ptr, float* y2_ptr, 
		float* prob_ptr, int* class_ptr, int* num_boxes, 
		int buffer_size, int width, int height, 
		int batch_size, float conf_thresh, float nms_thres);

private:
	double sigmoid(double x);
	vector<int> get_anchors(int net_grid);
	
	void parse_yolov4(ov::Tensor& output, float cof_threshold,
		int input_h, int input_w, int class_nums,
		vector<Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& o_class_pred);

	ov::Core core;
	std::shared_ptr<ov::Model> model;
	ov::CompiledModel compiled_model;

	string _inputname;
	string _onnxpath;
	float _cof_threshold;
	float _nms_IOU;
	int _class_nums;
	int _batch_size;
	//ģ������ͼƬ��С
	int _model_input_bs;
	int _model_input_w ;
	int _model_input_h ;
	int _model_input_c ;
	int _model_inputSize;
	int _model_input_nums;

	//ģ�����
	int _model_output_bs;
	int _model_output_c;
	int _model_output_w;
	int _model_output_h;
	int _model_output_class;
	int _model_output_nums;

	//ѡ�������豸
	string _device ;

	//ԭͼƬ��С
	int _org_h ;
	int _org_w ;

	int _buffer_size ;
	//ԭͼ������ͼ��ı���
	float _org_h_scale;
	float _org_w_scale ;
	//����ͼ������grid�ı���
	float _stride_w ;
	float _stride_h ;

};
