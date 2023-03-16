
#include"detector.h";

using namespace InferenceEngine;
using namespace std;
using namespace cv;
typedef unsigned char unit8_t;

std::string DEVICE = "GPU";
//开发主机

//测试机
std::string IR_filename_XML = "C:\\pic\\yolov4v1.xml";
std::string IR_filename_BIN = "C:\\pic\\yolov4v1.bin";

std::string IR_filename_ONNX = "C:\\pic\\yolov4v1.onnx";
//std::string IR_filename_ONNX = "C:\\keypoint\\model\\model.onnx";


string images_path = "C:\\pic\\pic\\";

int buff_size = 9999;
int height = 2048;
int width = 3072;
int batch_size = 1;
double conf_threshold = 0.5;  //置信度阈值,计算方法是框置信度乘以物品种类置信度
double nms_threshold = 0.01;  //nms最小重叠面积阈值

int cycle = 1;
//int input_w = 512;
//int input_h = 512;

int main(void)
{

	//1.建立对象
	Yolov4Dectector* detector = new Yolov4Dectector();
	
	//2.初始化模型
	//detector->InitializeDetector(DEVICE, IR_filename_XML);
	detector->InitializeDetector(DEVICE, IR_filename_ONNX);


	//3.图像前处理
	vector<cv::String> filenames;
	glob(images_path, filenames,false);
	int images_count = filenames.size();

	vector<Mat> images_all_list;
	vector<Mat> images_batch_list;

	//4.将所有读取的图片保存
	for (int i = 0; i < images_count; i++)
	{
		cv::Mat img = imread(filenames[i]);
		//width = images_batch_list[i].cols;
		//height = images_batch_list[i].rows;
		// 

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

			float* x1_ptr = new float[buff_size];
			float* y1_ptr = new float[buff_size];
			float* x2_ptr = new float[buff_size];
			float* y2_ptr = new float[buff_size];
			float* prob_ptr = new float[buff_size];
			int* class_ptr = new int[buff_size];
			int* num_boxes_ptr = new int[batch_size];
			int* num_boxes_ptr_copy = num_boxes_ptr;

			int per_pic_size = width * height * 3;

			unsigned char* img_batch = new unsigned char[per_pic_size * batch_size];

			//将batch中每张图片输入到图片指针0,1,2
			for (size_t batch_size_i = 0; batch_size_i < batch_size; batch_size_i++)
			{
				//将所有batch的图片加载到图片指针上
				memcpy(img_batch + batch_size_i * per_pic_size, images_batch_list[epoch_i * batch_size + batch_size_i].data, per_pic_size * sizeof(unsigned char));
			}


			//推理：对batch中每张图片进行推理和预测
			detector->process_frame(img_batch, x1_ptr, y1_ptr, x2_ptr, y2_ptr,
				prob_ptr, class_ptr, num_boxes_ptr, buff_size, width, height, batch_size,
				conf_threshold, nms_threshold);

			int num = 0;
			auto run_endtime = cv::getTickCount();
			auto infer_time = (to_string)((run_endtime - run_begintime) * 1000 / getTickFrequency() / batch_size);
			cout << "  +++++第" << epoch_i << "周期，共有" << batch_size << "张图片，单张图片的平均时间ms:" << infer_time << endl;

			//后处理：对batch中每个图片的结果进行解析
			for (size_t batch_size_i = 0; batch_size_i < batch_size; batch_size_i++)
			{
				//取出对应的图片
				Mat org_pic = images_all_list[epoch_i * batch_size + batch_size_i];


				//取出对应图片中的数据，有多少个batch，num_boxes_ptr中就有多少个数据
				//根据num_boxes_ptr每个数据值，从*x1_ptr，*y1_ptr等取得相应数据值个数的数据
				for (size_t box_num = 0; box_num < *num_boxes_ptr; box_num++)
				{
					float x1 = *(x1_ptr + num);
					float y1 = *(y1_ptr + num);
					float x2 = *(x2_ptr + num);
					float y2 = *(y2_ptr + num);
					float prob = *(prob_ptr + num);
					int label = *(class_ptr + num);
					num++;
					//Rect rect = Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
					Rect rect = Rect(*new Point2i(x1, y1), *new Point2i(x2, y2));

					//std::cout << "        第" << epoch_i <<"轮，第" << batch_size_i << "图片， 第" << box_num << "个框的x1:" << x1 << std::endl;
					//std::cout << "        第" << epoch_i << "轮，第" << batch_size_i << "图片，第" << box_num << "个框的y1:" << y1 << std::endl;
					//std::cout << "        第" << epoch_i << "轮，第" << batch_size_i << "图片，第" << box_num << "个框的x2:" << x2 << std::endl;
					//std::cout << "        第" << epoch_i << "轮，第" << batch_size_i << "图片，第" << box_num << "个框的y2:" << y2 << std::endl;
					//std::cout << "        第" << epoch_i << "轮，第" << batch_size_i << "图片，第" << box_num << "个框的prob:" << prob << std::endl;
					//std::cout << "        第" << epoch_i << "轮，第" << batch_size_i << "图片，第" << box_num << "个框的class:" << label << std::endl;
					//
					putText(org_pic, "class:" + to_string(label) + "  cof:" + to_string(prob), Point2f(x1, y1 - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
					rectangle(org_pic, rect, Scalar(255, 0, 0), 1, LINE_8);
				}
				num_boxes_ptr++;

				//输出图像
				//imshow("outputPIC", org_pic);
				cv::imwrite("C:\\pic\\result\\Result" + to_string(epoch_i) + "_" + to_string(batch_size_i) + ".jpg", org_pic);
			}

			//delete[] img_batch;
			//delete[] img_batch_copy;
			// 
			delete[] y1_ptr;
			delete[] x2_ptr;
			delete[] y2_ptr;
			delete[] prob_ptr;
			delete[] class_ptr;
			delete[] num_boxes_ptr_copy;
			delete[] img_batch;

		}
		cout << "      输出完成！" << endl;
	}


	waitKey(0);
	destroyAllWindows();
	return 0;
}