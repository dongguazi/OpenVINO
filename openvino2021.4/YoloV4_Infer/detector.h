#pragma once
#include<string>
#include<inference_engine.hpp>
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
	/// 初始化
	/// </summary>
	/// <param name="device"></param>
	/// <param name="xml_path"></param>
	/// <returns></returns>
	bool InitializeDetector(string device,string xml_path);

	bool uninit();

	/// <summary>
	/// 推理
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
	
	void parse_yolov4(const Blob::Ptr& blob, float cof_threshold,
		int input_h, int input_w, int class_nums,
		vector<Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& o_class_pred);

	ExecutableNetwork _netWork;
	InputsDataMap _inputsInfo;
	OutputsDataMap _outputinfo;

	string _inputname;
	string _xmlpath;
	float _cof_threshold;
	float _nms_IOU;
	int _class_nums;
	int _batch_size;
	//模型输入图片大小
	int _input_w ;
	int _input_h ;
	int _input_c ;
	int _inputSize;
	string _device ;

	//原图片大小
	int _org_h ;
	int _org_w ;

	int _buffer_size ;
	//原图和输入图像的比例
	float _org_h_scale;
	float _org_w_scale ;
	//输入图像和输出grid的比例
	float _stride_w ;
	float _stride_h ;

};
