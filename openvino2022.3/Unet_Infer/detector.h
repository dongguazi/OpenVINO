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
class UnetDectector
{
public:

	UnetDectector();

	~UnetDectector();


	/// <summary>
	/// 初始化
	/// </summary>
	/// <param name="device"></param>
	/// <param name="xml_path"></param>
	/// <returns></returns>
	bool InitializeDetector(string device, string xml_path);



	/// <summary>
	/// 推理过程
	/// </summary>
	/// <param name="image_batch"></param>
	/// <param name="prediction_batch"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="smallestMax"></param>
	/// <param name="batch_size"></param>
	void process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
		int width, int height, int smallestMax , int batch_size );

	bool uninit();
private:

	void parse_Unet(ov::Tensor& output, vector<Mat>& allPic);
	//void parse_Unet(ov::Tensor& output, unsigned char* result);

	void getResult_Unet(const float* output_buffer, vector<Mat>& cnts, int output_class, int output_w, int output_h);

	ov::Core core;
	std::shared_ptr<ov::Model> model;

	string _inputname;
	string _onnx_path;
	string _device;
	int _batch_size;
	//模型输入图片大小
	int _model_input_w;
	int _model_input_h;
	int _model_input_c;
	int _model_inputSize;


	int _model_output_w;
	int _model_output_h;
	int _model_output_class;

	//原图片大小
	int _org_h;
	int _org_w;

	int _buffer_size;
};
