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
class ClassifyClass
{
public:

	ClassifyClass();

	~ClassifyClass();


	/// <summary>
	/// 初始化
	/// </summary>
	/// <param name="device"></param>
	/// <param name="xml_path"></param>
	/// <returns></returns>
	bool InitializeDetector(string device, string xml_path);

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
	void process_frame(unsigned char* image_batch, float* prediction_batch,
		int width, int height, int training_size, int batch_size = 1);
	void process_frame2(unsigned char* image_batch, float* prediction_batch,
		int width, int height, int training_size, int batch_size);



private:

	void parse_classify(const Blob::Ptr& blob, vector<float>& allPic);

	ExecutableNetwork _netWork;
	InputsDataMap _inputsInfo;
	OutputsDataMap _outputinfo;

	string _inputname;
	string _onnxpath;
	int _batch_size;
	//模型输入图片大小
	int _input_w;
	int _input_h;
	int _input_c;
	int _inputSize;
	string _device;

	int _output_w;
	int _output_h;
	int _output_class;



	//原图片大小
	int _org_h;
	int _org_w;

	int _buffer_size;
	//原图和输入图像的比例
	float _org_h_scale;
	float _org_w_scale;
	//输入图像和输出grid的比例
	float _stride_w;
	float _stride_h;

};
