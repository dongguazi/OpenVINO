#include"detector.h";
#include"LicenseChecker.h"

UnetDectector::UnetDectector() {};
UnetDectector::~UnetDectector() {};

//std::string fileNameWithoutExt(const std::string& fpath) {
//	return fpath.substr(0, std::min<size_t>(fpath.size(), fpath.rfind('.')));
//}

/// <summary>
/// 初始化网络
/// </summary>
/// <param name="device">设备="CPU""GPU"</param>
/// <param name="xml_path">xml文件路径</param>
bool UnetDectector::InitializeDetector(string device, string xml_path)
{
	_xmlpath = xml_path;
	_device = device;
	cout << "*****0.初始化开始执行" << endl;

	//建立IEcore
	Core ie;
	CNNNetwork network;
	try
	{
		//_xmlpath可以为onnx或者xml类型
		network = ie.ReadNetwork(_xmlpath);
		cout << "*****1.读取模型文件" << endl;

	}
	catch (const std::exception& ex)
	{
		std::cout << "" << ex.what() << endl;
	}

	network.setBatchSize(1);

	auto devices = ie.GetAvailableDevices();
	cout << "*****2.加载当前运行设备为：" << device << endl;
	for (auto iter = devices.begin(); iter != devices.end(); iter++)
	{
		cout << "    可用设备名称1：" << iter->c_str() << endl;
	}

	//输入设置
	_inputsInfo = network.getInputsInfo();

	for (auto& item : _inputsInfo)
	{
		_inputname = item.first;
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
	}

	//输出设置
	_outputinfo = network.getOutputsInfo();
	for (auto& item : _outputinfo)
	{
		string imageOutputName = item.first;;
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
		//output_data->setLayout(Layout::NC);
	}
	//加载网络
	cout << "*****3.请等待，模型加载中........" << device << endl;

	try
	{
		auto load_begintime = cv::getTickCount();
		_netWork = ie.LoadNetwork(network, _device);
		auto load_endtime = cv::getTickCount();
		auto infer_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
		cout << "      *****模型加载时间时间ms:" << infer_time << endl;
		cout << "      模型加载完成！" << device << endl;
	}
	catch (const std::exception& ex)
	{
		std::cout << "" << ex.what() << endl;
		cout << "      模型加载失败！" << device << endl;
		return false;
	}

	return true;
}

/// <summary>
/// 推理过程，得到结果
/// </summary>
/// <param name="image_batch"></param>
/// <param name="prediction_batch"></param>
/// <param name="width"></param>
/// <param name="height"></param>
/// <param name="smallestMax"></param>
/// <param name="batch_size"></param>
void UnetDectector::process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
	int width, int height, int smallestMax = 512, int batch_size = 1)
{
	//获取原始输入参数
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;

	//建立推理请求
	auto infer_request = _netWork.CreateInferRequest();

	//解析图片信息，转为mat类型
	//std::vector<float> data;
	unsigned char* tempImage = image_batch;
	vector<Mat> org_pic;

	auto inputpic_begintime = cv::getTickCount();

	//解析每个batch中的图片
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3);
		sourceImage.data = tempImage + i * _org_h * _org_w * 3;
		org_pic.push_back(sourceImage);
	}

	auto outputpic_endtime = cv::getTickCount();
	auto infer_time = (to_string)((outputpic_endtime - inputpic_begintime) * 1000 / getTickFrequency());
	cout << "   ————输入图片转换时间:" << infer_time << endl;

	//string imageFilePath = "D:\\tf_train_my\\model\\20210722190514412.bmp";
	//Mat img = imread(imageFilePath);
	//string  output_csv = "D:\\tf_train_my\\output.xls";

	int count_pic = 0;
	int num = 0;
	vector<Mat> batch_Pic;

	//循环所有图片，进行推理预测
	for (auto pic : org_pic)
	{
		count_pic++;
		cout << endl;
		cout << "   —+—+—+—开始循环第" << count_pic << "张图片" << endl;

		Mat img = pic.clone();

		//循环获取input——blob，一般输入只有一个
		for (auto item : _inputsInfo)
		{
			auto inpublob_begintime = cv::getTickCount();

			auto input_name = item.first;
			auto  framBlob = infer_request.GetBlob(input_name);

			_input_c = framBlob->getTensorDesc().getDims()[1];
			_input_h = framBlob->getTensorDesc().getDims()[2];
			_input_w = framBlob->getTensorDesc().getDims()[3];

			Mat blob_image;
			//将原始图片转换为输入图片大小
			resize(img, blob_image, Size(_input_w, _input_h));
			//转换BGR到RGB
			cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

			auto inpublob_endtime = cv::getTickCount();
			auto infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			cout << "        ————图片resize时间:" << infer_time << endl;

			inpublob_begintime = cv::getTickCount();

			//更新input_blob数据
			float* blob_data = static_cast<float*>(framBlob->buffer());

			for (size_t c = 0; c < _input_c; c++)
			{
				for (size_t h = 0; h < _input_h; h++)
				{
					for (size_t w = 0; w < _input_w; w++)
					{
						blob_data[c * _input_h * _input_w + h * _input_w + w] = (float)(blob_image.at<cv::Vec3b>(h, w)[c]) / 255.0f;
					}
				}
			}

			inpublob_endtime = cv::getTickCount();
			infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			cout << "        ————输入blob转换时间:" << infer_time << endl;
		}

		//对图片进行推理
		auto infer_begintime = cv::getTickCount();
		//同步推理
		//infer_request.Infer();
		//异步推理
		infer_request.StartAsync();
		infer_request.Wait(InferRequest::WaitMode::RESULT_READY);

		auto infer_endtime = cv::getTickCount();
		infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		cout << "   ——————————————————————————————第" << count_pic << "张图片的异步推理时间:" << infer_time << endl;

		////加密狗
		//char* code = CheckLicense();
		//cout << code << endl;
		//if (code == NULL)
		//{
		//	throw exception("License Error");
		//}
		//std::string name = string(code);

		//cout << "name" << name << endl;
		//size_t postion = name.find("VxDeepVino");

		//if (postion == string::npos)
		//	throw exception("License Error");
				
		//输出
		auto output_begintime= cv::getTickCount();

		//获取输出结果
		for (auto item : _outputinfo)
		{
			//1.获取 blob
			auto output_name = item.first;
			auto output_blob = infer_request.GetBlob(output_name);

			//输出解析cnts
            //1.第一种直接输出指针
			parse_Unet(output_blob, batch_Pic);
		}

		auto output_endtime = cv::getTickCount();
		infer_time = (to_string)((output_endtime - output_begintime) * 1000 / getTickFrequency());
		cout << "   ————第" << count_pic << "张图片的输出处理时间:" << infer_time << endl;
	}

	//最终输出  _org_h*_org_w * _output_class* _batch_size
	int picSzie_per = _org_h * _org_w;
	int batch_size_class_per = picSzie_per * _output_class;

	//unsigned char* result = new unsigned char[batch_size_class_per * _batch_size];
	//将batch中每张图片输入到图片指针0,1,2
	for (size_t batchsize_i = 0; batchsize_i < _batch_size; batchsize_i++)
	{
		for (size_t pic_i = 0; pic_i < _output_class; pic_i++)
		{
			//将所有batch的图片加载到图片指针上
			memcpy(prediction_batch + batchsize_i * batch_size_class_per + picSzie_per * pic_i, batch_Pic[batchsize_i * _output_class + pic_i].data, picSzie_per * sizeof(unsigned char));
		}
	}
}


void UnetDectector::process_frame2(unsigned char* image_batch, unsigned char* prediction_batch,
	int width, int height, int smallestMax = 512, int batch_size = 1)
{
	//获取原始输入参数
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;

	//建立推理请求
	auto infer_request = _netWork.CreateInferRequest();



	auto inputpic_begintime = cv::getTickCount();

	auto outputpic_endtime = cv::getTickCount();
	auto infer_time = (to_string)((outputpic_endtime - inputpic_begintime) * 1000 / getTickFrequency());
	cout << "   ————输入图片转换时间:" << infer_time << endl;


	int count_pic = 0;
	int num = 0;
	vector<Mat> batch_Pic;

	//循环所有图片，进行推理预测
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3, image_batch + i * _org_h * _org_w * 3);

		count_pic++;
		//cout << endl;
		//cout << "   —+—+—+—开始循环第" << count_pic << "张图片" << endl;

		Mat img = sourceImage.clone();

		//循环获取input——blob，一般输入只有一个
		for (auto item : _inputsInfo)
		{
			auto inpublob_begintime = cv::getTickCount();

			auto input_name = item.first;
			auto  framBlob = infer_request.GetBlob(input_name);

			_input_c = framBlob->getTensorDesc().getDims()[1];
			_input_h = framBlob->getTensorDesc().getDims()[2];
			_input_w = framBlob->getTensorDesc().getDims()[3];

			Mat blob_image;
			//将原始图片转换为输入图片大小
					//将原始图片转换为输入图片大小
			cv::resize(sourceImage, sourceImage, cv::Size(_input_w, _input_h));
			//转换BGR到RGB
			//cvtColor(blob_image, blob_image, COLOR_BGR2RGB);
			sourceImage.convertTo(sourceImage, CV_32FC3, 1 / 255.0f);
			
			auto inpublob_endtime = cv::getTickCount();
			auto infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			//cout << "        ————图片resize时间:" << infer_time << endl;

			inpublob_begintime = cv::getTickCount();

			//更新input_blob数据
			float* blob_data = static_cast<float*>(framBlob->buffer());

			std::vector<cv::Mat>channles(3);
			cv::split(sourceImage, channles);
			memcpy(blob_data + i * 3 * _input_h * _input_w, channles[0].data, _input_h * _input_w * sizeof(float));
			memcpy(blob_data + i * 3 * _input_h * _input_w + _input_h * _input_w, channles[1].data, _input_h * _input_w * sizeof(float));
			memcpy(blob_data + i * 3 * _input_h * _input_w + 2 * _input_h * _input_w, channles[2].data, _input_h * _input_w * sizeof(float));

			//for (size_t c = 0; c < _input_c; c++)
			//{
			//	for (size_t h = 0; h < _input_h; h++)
			//	{
			//		for (size_t w = 0; w < _input_w; w++)
			//		{
			//			blob_data[c * _input_h * _input_w + h * _input_w + w] = (float)(blob_image.at<cv::Vec3b>(h, w)[c]) / 255.0f;
			//		}
			//	}
			//}

			inpublob_endtime = cv::getTickCount();
			infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			//cout << "        ————输入blob转换时间:" << infer_time << endl;
		}

		//对图片进行推理
		auto infer_begintime = cv::getTickCount();
		//同步推理
		//infer_request.Infer();
		//异步推理
		infer_request.StartAsync();
		infer_request.Wait(InferRequest::WaitMode::RESULT_READY);

		auto infer_endtime = cv::getTickCount();
		infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		cout << "第" << count_pic << "张图片的异步推理时间ms:" << infer_time << endl;

		////加密狗
		//char* code = CheckLicense();
		//cout << code << endl;
		//if (code == NULL)
		//{
		//	throw exception("License Error");
		//}
		//std::string name = string(code);

		//cout << "name" << name << endl;
		//size_t postion = name.find("VxDeepVino");

		//if (postion == string::npos)
		//	throw exception("License Error");

		//输出
		auto output_begintime = cv::getTickCount();

		//获取输出结果
		for (auto item : _outputinfo)
		{
			//1.获取 blob
			auto output_name = item.first;
			auto output_blob = infer_request.GetBlob(output_name);

			//输出解析cnts
			//1.第一种直接输出指针
			parse_Unet(output_blob, batch_Pic);
		}

		auto output_endtime = cv::getTickCount();
		infer_time = (to_string)((output_endtime - output_begintime) * 1000 / getTickFrequency());
		//cout << "   ————第" << count_pic << "张图片的输出处理时间:" << infer_time << endl;
	}

	//最终输出  _org_h*_org_w * _output_class* _batch_size
	int picSzie_per = _org_h * _org_w;
	int batch_size_class_per = picSzie_per * _output_class;

	//unsigned char* result = new unsigned char[batch_size_class_per * _batch_size];
	//将batch中每张图片输入到图片指针0,1,2
	for (size_t batchsize_i = 0; batchsize_i < _batch_size; batchsize_i++)
	{
		for (size_t pic_i = 0; pic_i < _output_class; pic_i++)
		{
			//将所有batch的图片加载到图片指针上
			memcpy(prediction_batch + batchsize_i * batch_size_class_per + picSzie_per * pic_i, batch_Pic[batchsize_i * _output_class + pic_i].data, picSzie_per * sizeof(unsigned char));
		}
	}
}

/// <summary>
/// 解析output输出最后结果
/// </summary>
/// <param name="blob">输入blob</param>
/// <param name="cnts">输出uchar*</param>
/// <param name="width">原图宽度</param>
/// <param name="height">原图高度</param>
void UnetDectector::parse_Unet(const Blob::Ptr& blob, vector<Mat>& allPic)
{
	//1.获取输出的dims
	auto outputDims = blob->getTensorDesc().getDims();

	//2.获得图片尺寸，
	int output_class = outputDims[1];
	_output_class = output_class - 1;
	_output_h= outputDims[2];
	_output_w = outputDims[3];

	//3.获取blob数据
	LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
	float* output_buffer = blobMapped.as<float*>();

	UnetDectector::getResult_Unet(output_buffer, allPic, output_class, _output_w, _output_h);
	
}

/// <summary>
/// 获得blob最后内容uchar*
/// </summary>
/// <param name="output_buffer">const float* 输入</param>
/// <param name="cnts">uchar*输出</param>
/// <param name="output_class">分类数</param>
/// <param name="output_w">模型输出宽度</param>
/// <param name="output_h">模型输出高度</param>
void UnetDectector::getResult_Unet(const float* output_buffer, vector<Mat>& allPic, int output_class, int output_w, int output_h)
{
	int size = output_w * output_h;
	int nums = output_class * size;
	uchar* cnts =new unsigned char[nums];
	uchar* newcnts = cnts;

	//for (size_t i = size; i < nums; i++)
	//{
	//	uchar pixel = *(output_buffer + i) * 255;
	//	*cnts = pixel;
	//	cnts++;
	//}
	for (size_t i = size; i < nums; i++)
	{
		uchar pixel = *(output_buffer + i) * 255;
		*newcnts = pixel;
		newcnts++;
	}

	//输出解析

	int output_size = _org_w * _org_h;
	int output_class_last = output_class - 1;
	for (size_t i_class = 0; i_class < output_class_last; i_class++)
	{
		//Mat resultImage = Mat(output_h, output_w, CV_8UC1, Scalar(0));
		vector <uchar> mask;
		mask.resize(size);
		cout << "图片：" << i_class << endl;

		for (size_t h = 0; h < size; h++)
		{
			mask[h] = 0;
			float pixel = *(cnts + size * i_class + h);
			mask[h] = pixel;
		}
		unsigned char* maskImage = mask.data();
		cv::Mat mask_mat = cv::Mat(output_h, output_w, CV_8UC1);
		mask_mat.data = maskImage;

		cv::Mat mask_mat_resized;
		cv::resize(mask_mat, mask_mat_resized, cv::Size(_org_w, _org_h));

		allPic.push_back(mask_mat_resized);
	}
	int test = false;
	if (test)
	{
		//打印验证结果
		int num = 0;
		for (auto pic : allPic)
		{
			num++;
			stringstream ss;
			ss << num;
			imwrite(ss.str() + ".jpg", pic);
		}

	}

	delete[] cnts;	
}



bool UnetDectector::uninit() {
	return true;
}




