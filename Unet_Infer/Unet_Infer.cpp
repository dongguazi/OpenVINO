﻿
#include"detector.h";

using namespace InferenceEngine;
using namespace std;
using namespace cv;
typedef unsigned char unit8_t;

std::string DEVICE = "CPU";
std::string IR_filename_IR = "..\\model\\segment\\model.xml";

//std::string IR_filename_ONNX = "D:\\OpenVINO\\model\\UnetLane_mutilLane_ZQ.onnx";
//string images_path = "D:\\OpenVINO\\model\\segment\\pic";
std::string IR_filename_ONNX = "D:\\OpenVINO\\model\\segment2\\model.onnx";
string images_path = "D:\\OpenVINO\\model\\segment2\\pic";

//需要根据实际情况设置的参数
int buff_size = 9999;
int buff_size_Out = 99999999;

int width = 1292;
int height = 964;

int batch_size = 1;

int cycle = 1;

int main(void)
{

	//Mat sourceImage = imread("G:\\segment0\\pic\\20230612103130726.jpg",1);
	//for (int ii = 0; ii < width * height * 3 * batch_size; ii++)
	//{
	//	if ((float)(sourceImage.data[ii]) > 200)
	//	{
	//		std::cout << ii << ": " << (float)(sourceImage.data[ii]) << std::endl;
	//	}
	//}

	//1.建立对象
	UnetDectector* detector = new UnetDectector();

	//2.初始化模型
	detector->InitializeDetector(DEVICE, IR_filename_ONNX);

	//3.图像前处理
	vector<cv::String> filenames;
	glob(images_path, filenames, false);
	int images_count = filenames.size();

	vector<Mat> images_all_list;
	vector<Mat> images_batch_list;
	bool isFirst = true;
	//4.将所有读取的图片保存
	for (int i = 0; i < images_count; i++)
	{
		cv::Mat img = imread(filenames[i]);
		//width = images_batch_list[i].cols;
		//height = images_batch_list[i].rows;
		// 
		if (isFirst)
		{
			width = img.cols;
			height = img.rows;
			isFirst = false;
		}
		images_all_list.push_back(img);
	}

	int epoch = images_all_list.size() / batch_size;

	//5.按照batch大小分割成epoch
	for (size_t i = 0; i < epoch; i++)
	{
		for (size_t j = 0; j < batch_size; j++)
		{
			Mat img = images_all_list[i * batch_size + j].clone();
			images_batch_list.push_back(img);
		}
	}

	buff_size_Out = batch_size * height * width*6;

	cout << "*****4.开始推理" << endl;

	//6.输入所有，进行预测输出所有结果
	  //循环每个epoch，每次都输出预测的结果
	for (size_t c = 0; c < cycle; c++)
	{
		cout << "~~~~~~~~~~~~~~~~~~内存泄漏循环次数：" << c << "~~~~~~~~~~~~~~~~~~~~~~" << endl;
		for (size_t epoch_i = 0; epoch_i < epoch; epoch_i++)
		{
			cout << endl;
			cout << "   +++++开始第" << epoch_i << "周期：" << endl;

			auto run_begintime = cv::getTickCount();

			int per_pic_size = width * height * 3;

			unsigned char* img_batch = new unsigned char[per_pic_size * batch_size];

			//将batch中每张图片输入到图片指针0,1,2
			for (size_t batch_size_i = 0; batch_size_i < batch_size; batch_size_i++)
			{
				//cv::imwrite("org.jpg", images_batch_list[epoch_i * batch_size + batch_size_i]);
				//将所有batch的图片加载到图片指针上
				memcpy(img_batch + batch_size_i * per_pic_size, images_batch_list[epoch_i * batch_size + batch_size_i].data, per_pic_size * sizeof(unsigned char));
			}

			unsigned char* prediction_batch = new unsigned char[buff_size_Out];

			//推理：对batch中每张图片进行推理和预测
			//void process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
			//	int width, int height, int smallestMax = 512, int batch_size = 1);
			detector->process_frame(img_batch, prediction_batch,width,height,512, batch_size);

			int num = 0;
			auto run_endtime = cv::getTickCount();
			auto infer_time = (to_string)((run_endtime - run_begintime) * 1000 / getTickFrequency() / batch_size);
			cout << "  +++++第" << epoch_i << "周期，共有" << batch_size << "张图片，单张图片的平均时间ms:" << infer_time << endl;
		
			delete[] img_batch;
			delete[] prediction_batch;
		}
		cout << "      输出完成！" << endl;
	}

	delete detector;
	waitKey(0);
	destroyAllWindows();
	return 0;
}