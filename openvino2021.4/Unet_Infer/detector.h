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
class UnetDectector
{
public:

	UnetDectector();

	~UnetDectector();


	/// <summary>
	/// ��ʼ��
	/// </summary>
	/// <param name="device"></param>
	/// <param name="xml_path"></param>
	/// <returns></returns>
	bool InitializeDetector(string device, string xml_path);



	/// <summary>
	/// �������
	/// </summary>
	/// <param name="image_batch"></param>
	/// <param name="prediction_batch"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="smallestMax"></param>
	/// <param name="batch_size"></param>
	void process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
		int width, int height, int smallestMax , int batch_size );

	void process_frame2(unsigned char* image_batch, unsigned char* prediction_batch,
		int width, int height, int smallestMax, int batch_size);

	bool uninit();
private:

	void parse_Unet(const Blob::Ptr& blob, vector<Mat>& allPic);

	void getResult_Unet(const float* output_buffer, vector<Mat>& cnts, int output_class, int output_w, int output_h);



	ExecutableNetwork _netWork;
	InputsDataMap _inputsInfo;
	OutputsDataMap _outputinfo;

	string _inputname;
	string _xmlpath;
	int _batch_size;
	//ģ������ͼƬ��С
	int _input_w;
	int _input_h;
	int _input_c;
	int _inputSize;
	string _device;

	int _output_w;
	int _output_h;
	int _output_class;



	//ԭͼƬ��С
	int _org_h;
	int _org_w;

	int _buffer_size;
	//ԭͼ������ͼ��ı���
	float _org_h_scale;
	float _org_w_scale;
	//����ͼ������grid�ı���
	float _stride_w;
	float _stride_h;

};
