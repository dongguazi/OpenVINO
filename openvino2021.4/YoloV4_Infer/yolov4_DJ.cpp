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
void frameToBlob(const cv::Mat& frame, InferRequest& inferRequest, const std::string& inputName)
{
	Blob::Ptr frameBlob = inferRequest.GetBlob(inputName);
	matU8ToBlob<unit8_t>(frame, frameBlob);
	//还有一种转换的方式，输入的时cv：：matde 格式，输出resulting Blob pointer（Blob::Ptr）
	//wrapMat2Blob(const cv::Mat& mat);
}
Blob::Ptr mat2Blob(const cv::Mat& frame)
{
	return wrapMat2Blob(frame);
}
std::string DEVICE = "GPU";
//std::string IR_filename_XML = "D:\\tf_train_my\\opencv_tutorial_data-master\\models\\yolov5\\yolov5s.xml";
//std::string IR_filename_BIN = "D:\\tf_train_my\\opencv_tutorial_data-master\\models\\yolov5\\yolov5s.bin";
std::string IR_filename_XML = "D:\\tf_train_my\\model\\yolov4v1.xml";
std::string IR_filename_BIN = "D:\\tf_train_my\\model\\yolov4v1.bin";



string imageFile = "D:\\tf_train_my\\bus.jpg";
//string imageFile_test = "D:\\tf_train\\workspaces\\cats_dogs\\images\\test\\4.jpg";

double confidence_threshold = 0.99;  //置信度阈值,计算方法是框置信度乘以物品种类置信度
double nms_area_threshold = 0.99;  //nms最小重叠面积阈值

vector<string> labels = { "fake","cat","dog" };
/// <summary>
/// 获得不同规格的锚框
/// </summary>
/// <param name="net_grid"></param>
/// <returns></returns>
vector<int> get_anchors(int net_grid)
{
	vector<int> anchors(6);
	int achors80[6] = { 10,13,16,30,33,23 };
	int achors40[6] = { 30,61, 62,45, 59,119 };
	int achors20[6] = { 116,90, 156,198, 373,326 };
	if (net_grid == 80)
	{
		anchors.insert(anchors.begin(), achors80, achors80 + 6);

	}
	else if (net_grid == 40)
	{
		anchors.insert(anchors.begin(), achors40, achors40 + 6);

	}
	else if (net_grid == 20)
	{
		anchors.insert(anchors.begin(), achors20, achors20 + 6);

	}
	return anchors;
}

/// <summary>
/// sigmoid函数
/// </summary>
/// <param name="x"></param>
/// <returns></returns>
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}


int main(void)
{

	//0.输出inferenceEgine当前版本号
	cout << "inferenceEgine:" << GetInferenceEngineVersion() << endl;


	//1.创建core模型
	cout << "1.创建core模型。。。" << endl;

	Core ie;

	auto devicesVersions = ie.GetVersions(DEVICE);
	for (auto iter = devicesVersions.rbegin(); iter != devicesVersions.rend(); iter++)
	{
		cout << "    设备名称：" << iter->first << "，设备版本号：" << iter->second << endl;
	}
	auto devices = ie.GetAvailableDevices();
	for (auto iter = devices.begin(); iter != devices.end(); iter++)
	{
		cout << "    可用设备名称1：" << iter->c_str() << endl;

	}
	//for (int i = 0; i < devices.size(); i++)
	//{
	//	cout << "可用设备名称2：" << devices[i] << endl;
	//}


	//2.读取IR模型文件
	cout << "2.读取IR模型。。。" << endl;
	clock_t begin1;
	clock_t end1;
	begin1 = clock();

	CNNNetwork network;

	try
	{
		network = ie.ReadNetwork(IR_filename_XML);

		//network = ie.ReadNetwork(ONNX_file);
	}
	catch (const std::exception& ex)
	{
		std::cout << "" << ex.what() << endl;
	}

	end1 = clock();
	network.setBatchSize(1);

	cout << "    加载时间：" << (to_string)(end1 - begin1) << "ms" << endl;
	cout << "    读取IR模型完成！" << endl;

	//3.配置输入输出
	cout << "3.配置输入输出。。。" << endl;
	string imageInputName, imInfoInputName;
	//InputInfo::Ptr input_data = nullptr; 

	string imageOutputName, imInfoOutputName;
	//InputInfo::Ptr output_data = nullptr;

	//输入设置
	InputsDataMap inputsInfo = network.getInputsInfo();

	for (auto& item : inputsInfo)
	{
		imageInputName = item.first;
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);

	}
	//输出设置
	OutputsDataMap outputsInfo = network.getOutputsInfo();
	for (auto& item : outputsInfo)
	{
		imageOutputName = item.first;
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
		//output_data->setLayout(Layout::NC);
	}
	cout << "    配置输入输出完成！" << endl;


	//4.加载模型
	cout << "4.加载模型。。。" << endl;
	cout << "    开始加载模型！" << endl;

	clock_t loadtimebegin;
	clock_t loadtimeend;
	loadtimebegin = clock();
	auto executable_network = ie.LoadNetwork(network, DEVICE);
	loadtimeend = clock();
	cout << "    加载时间：" << (to_string)(loadtimeend - loadtimebegin) << "ms" << endl;

	cout << "    加载模型完成！" << endl;


	//5.推理请求
	cout << "5.创建推理请求。。。" << endl;
	auto infer_request = executable_network.CreateInferRequest();
	cout << "    创建请求完成！" << endl;


	//6.输入数据处理
	//将opencv中的mat对象转换为inferenceEngineBlob对象，获得图像数据
	cout << "6.准备输入数据。。。" << endl;
	Mat img = cv::imread(imageFile);
	if (img.empty())
	{
		cout << "      无效图片输入" << endl;
	}
	Mat img_copy = img.clone();
	cout << "       原图通道数：" << img_copy.channels() << endl;

	resize(img_copy, img_copy, Size(640, 640));


	auto width = img_copy.cols;
	auto height = img_copy.rows;

	//imshow("输入图像", img);  

	for (auto& item : inputsInfo)
	{
		auto input_name = item.first;
		auto  input = infer_request.GetBlob(input_name);

		cout << "     输入blob的个数" << input->byteSize() << endl;;
		//cout << "输入blob的个数" << input->buffer() << endl;;


		//方法1：利用frameToBlob，直接修改infer_request的buffer数据
		//frameToBlob(img_copy, infer_request, input_name);


		//方法2：利用wrapMat2Blob，返回数据再传进input_name层
		auto data_blob = wrapMat2Blob(img_copy);
		infer_request.SetBlob(input_name, data_blob);


		//方法3：自己写程序处理，将mat转变为blob::Ptr

#pragma region MyRegion
//获得模型的CHW
		//size_t num_channel = input->getTensorDesc().getDims()[1];
		//size_t h = input->getTensorDesc().getDims()[2];
		//size_t w = input->getTensorDesc().getDims()[3];

		//size_t image_size = h * w;
		//Mat blob_image;
		//// 总结：模型中顺序(h,w)，channel为RGB，opencv中顺序(rows,cols)即(h,w),channel为BGR,
		//// 将输入的图像转变为模型可以接受的尺寸resize
		//resize(img_copy, blob_image, Size(w, h));

		////opencv读取的图像channel是BGR，需要转换到模型可用的RGB
		//cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

		////将RGB图像转换为推理可用的blob::Ptr的向量，并写入到推理器
		////data 直接指向了buffer的内存地址，修改data可以直接修改infer中buffer内的值(blob::Ptr)
		//cout << "原图通道数：" << blob_image.channels() << endl;

		////float* data = static_cast<float*>(input->buffer());
		//InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(input)->wmap();
		//float* blob_data = blobMapped.as<float*>();
		//


		//for (size_t row = 0; row < h; row++)
		//{
		//	for (size_t col = 0; col < w; col++)
		//	{
		//		for (size_t ch = 0; ch < num_channel; ch++)
		//		{
		//			float a = float(blob_image.at <Vec3b>(row, col)[ch] / 255.0f);
		//			blob_data[image_size * ch + row * w + col] = a;

		//		}
		//	}
		//}
#pragma endregion	
	}

	cout << "    输入数据完成！" << endl;


	//7.推理
	cout << "7.推理。。。" << endl;
	clock_t begin, end;
	begin = clock();
	auto begintime = cv::getTickCount();
	//Mat image1 = imread(imageFile_test);

	//frameToBlob(image1, infer_request,imageInputName);

	infer_request.StartAsync();
	infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);
	end = clock();

	auto endtime = cv::getTickCount();

	//auto infer_time = (to_string)((double)(end - begin));
	//auto infer_time = (to_string)((endtime - begintime) * 1000 / getTickFrequency());

	//infer_time << "推理时间：" << (double)(end - begin) << "ms";
	//putText(img, "time:" + infer_time + "ms", Point2f(0, 12), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0, 255, 0));

	cout << "    推理完成！" << endl;


	//8.输出
	cout << "8.输出。。。" << endl;

	//获取各层结果
	vector<Rect> origin_rect;
	vector<float> origin_rect_cof;

	begintime = cv::getTickCount();

	//auto detections = infer_request.GetBlob(imageOutputName);
	for (auto item : outputsInfo)
	{
		//1.获取 blob
		auto output_name = item.first;
		auto output = infer_request.GetBlob(output_name);

		//2.获取输出的dims
		auto outputDims = output->getTensorDesc().getDims();
		cout << output_name << "    输出维度维度:" << outputDims.size() << endl;
		//们曾输出的格式为：n，layers，h,w,out
		// out[dim=4]=85,为offset_x，offset_y,scale_h,scale_w,obj_confid,probs=80
		//第一个输出尺寸417：1,3,40,40,85
		//第二个输出尺寸437：1，3，20，20，85
		//第三个输出尺寸output：1，3，80，80，85 	

		//获得图片尺寸，计算相应的anchors，每个中心点有三个anchors
		int net_grid = outputDims[2];
		vector<int> anchors = get_anchors(net_grid);

		//获取结果输出的维度
		int net_output_size = outputDims[4];
		size_t anchor_n = 3;

		//3.获取blob数据指针
	/*	auto const memLocker = output->cbuffer();
		const float* output_buffer = memLocker.as <const float*>();*/

		LockedMemory<const void> blobMapped = as<MemoryBlob>(output)->rmap();
		const float* output_buffer = blobMapped.as<float*>();
		//4.计算置信度cofid和rect
		for (int n = 0; n < anchor_n; ++n)
			for (int i = 0; i < net_grid; ++i)
				for (int j = 0; j < net_grid; ++j)
				{
					int object_index = n * net_grid * net_grid * net_output_size + i * net_grid * net_output_size + j * net_output_size;

					/*			for (int i = 0; i < 15; i++)
								{
									cout << "   输出：" <<to_string(i) << "==" << output_buffer[object_index + i]<<endl;
								}*/
								//每张图
					double box_prob = output_buffer[object_index + 4];
					box_prob = sigmoid(box_prob);
					//当前置信度不满足总体置信度
					if (box_prob < confidence_threshold)
					{
						continue;
					}
					//此处输出为中心点坐标，要换为交点坐标
					double x = output_buffer[object_index + 0];
					double y = output_buffer[object_index + 1];
					double w = output_buffer[object_index + 2];
					double h = output_buffer[object_index + 3];

					double max_prob = 0;
					int idx = 0;
					for (int t = 5; t < net_output_size; ++t)
					{
						//找到80个类别中得分最高的类别
						double tp = output_buffer[object_index + t];
						tp = sigmoid(tp);
						if (tp > max_prob)
						{
							max_prob = tp;
							idx = t;
						}
					}
					float cof = box_prob * max_prob;
					if (cof < confidence_threshold)
					{
						continue;
					}
					//
					x = (sigmoid(x) * 2 - 0.5 + j) * 640.0f / net_grid;
					y = (sigmoid(y) * 2 - 0.5 + i) * 640.0f / net_grid;
					w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
					h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

					double r_x = x - w / 2;
					double r_y = y - h / 2;
					Rect rect = Rect(round(r_x), round(r_y), round(w), round(h));
					origin_rect.push_back(rect);
					origin_rect_cof.push_back(cof);
				}
	}


	//9.计算最终结果
	vector<int> final_id;
	dnn::NMSBoxes(origin_rect, origin_rect_cof, confidence_threshold, nms_area_threshold, final_id);

	for (int i = 0; i < final_id.size(); ++i)
	{
		Rect resize_rect = origin_rect[final_id[i]];
		double cof = origin_rect_cof[final_id[i]];
		cout << "     得分：" << to_string(cof) << endl;

		putText(img_copy, to_string(cof), Point2f(resize_rect.x, resize_rect.y - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
		rectangle(img_copy, resize_rect, Scalar(255, 0, 0), 1, LINE_8);

	}
	//10.计算推理时间
	endtime = cv::getTickCount();
	auto infer_time = (to_string)((endtime - begintime) * 1000 / getTickFrequency());
	putText(img_copy, "time:" + infer_time + "ms", Point2f(0, 12), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0, 255, 0));


	cout << "      输出完成！" << endl;

	cout << "9.全部完成，输出结果图！" << endl;

	imshow("检测结果", img_copy);
	waitKey(0);
	destroyAllWindows();
	return 0;
}