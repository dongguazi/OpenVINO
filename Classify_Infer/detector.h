#pragma once
#include<string>
#include<ie/inference_engine.hpp>
#include<samples/ocv_common.hpp>
#include<ngraph/ngraph.hpp>
#include<opencv2/opencv.hpp>
#include<time.h>

#include <openvino/openvino.hpp>

using namespace InferenceEngine;
using namespace std;
using namespace cv;
typedef unsigned char unit8_t;
class ClassifyClass
{
public:

	ClassifyClass();

	~ClassifyClass();

	bool InitializeDetector(string device, string xml_path);

	bool uninit();

	void process_frame(unsigned char* image_batch, float* prediction_batch,
		int width, int height, int training_size, int batch_size = 1);
	void process_frame_batch(unsigned char* image_batch, float* prediction_batch,
		int width, int height, int training_size, int batch_size = 1);

private:

	void parse_classify(ov::Tensor& output, vector<float>& per_pic_res);
	ov::Core core;
	std::shared_ptr<ov::Model> model;
	ov::CompiledModel compiled_model;
	string mode_type;
	string _onnxpath;
	int _batch_size;

	//模型输入
	int _model_input_bs;
	int _model_input_w;
	int _model_input_h;
	int _model_input_c;
	int _model_inputSize;
	int _model_input_nums;


	//模型输出
	int _model_output_bs;
	int _model_output_c;
	int _model_output_w;
	int _model_output_h;
	int _model_output_class;
	int _model_output_nums;
	
	//选择运行设备
	string _device;


	//原图片大小
	int _org_h;
	int _org_w;

};
