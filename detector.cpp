#include"detector.h";
#include"LicenseChecker.h"

ClassifyClass::ClassifyClass() {};
ClassifyClass::~ClassifyClass() {};

//std::string fileNameWithoutExt(const std::string& fpath) {
//	return fpath.substr(0, std::min<size_t>(fpath.size(), fpath.rfind('.')));
//}

/// <summary>
/// ��ʼ������
/// </summary>
/// <param name="device">�豸="CPU""GPU"</param>
/// <param name="xml_path">xml�ļ�·��</param>
bool ClassifyClass::InitializeDetector(string device, string xml_path)
{
	_onnxpath = xml_path;
	//cout << "onnx_path" << _onnxpath << endl;
	_device = device;
	//cout << "*****0.��ʼ����ʼִ��" << endl;

	//����IEcore
	Core ie;
	CNNNetwork network;
	try
	{
		//string binfile= fileNameWithoutExt(_xmlpath) + ".bin";
		network = ie.ReadNetwork(_onnxpath);
		
		//cout << "*****1.��ȡģ���ļ�" << endl;

	}
	catch (const std::exception& ex)
	{
		//std::cout << "" << ex.what() << endl;
	}

	network.setBatchSize(1);

	auto devices = ie.GetAvailableDevices();
	//cout << "*****2.���ص�ǰ�����豸Ϊ��" << device << endl;
	for (auto iter = devices.begin(); iter != devices.end(); iter++)
	{
		//cout << "    �����豸����1��" << iter->c_str() << endl;
	}

	//��������
	_inputsInfo = network.getInputsInfo();

	for (auto& item : _inputsInfo)
	{
		_inputname = item.first;
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
		//input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
	}

	//�������
	_outputinfo = network.getOutputsInfo();
	for (auto& item : _outputinfo)
	{
		string imageOutputName = item.first;;
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
		//output_data->setLayout(Layout::NC);
	}
	//��������
	//cout << "*****3.��ȴ���ģ�ͼ�����........" << device << endl;

	try
	{
		auto load_begintime = cv::getTickCount();
		_netWork = ie.LoadNetwork(network, _device);
		auto load_endtime = cv::getTickCount();
		auto infer_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
		cout << "*ģ�ͼ��سɹ���" << device << endl;
		cout << "*ģ�ͼ���ʱ��ʱ��ms:" << infer_time << endl;

	}
	catch (const std::exception& ex)
	{
		//std::cout << "" << ex.what() << endl;
		cout << "*ģ�ͼ���ʧ�ܣ�" << device << endl;
		return false;
	}

	return true;
}

bool ClassifyClass::uninit() {
	return true;
}


void ClassifyClass::process_frame(unsigned char* image_batch, float* prediction_batch,
	int width, int height, int training_size, int batch_size)
{
	//��ȡԭʼ�������
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;

	//������������
	auto infer_request = _netWork.CreateInferRequest();

	//����ͼƬ��Ϣ��תΪmat����
	//std::vector<float> data;
	//unsigned char* tempImage = image_batch;
	//vector<Mat> org_pic;

	auto inputpic_begintime = cv::getTickCount();

	////����ÿ��batch�е�ͼƬ
	//for (int i = 0; i < batch_size; i++)
	//{
	//	cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3);
	//	sourceImage.data = tempImage + i * _org_h * _org_w * 3;
	//	org_pic.push_back(sourceImage);
	//}

	auto outputpic_endtime = cv::getTickCount();
	auto infer_time = (to_string)((outputpic_endtime - inputpic_begintime) * 1000 / getTickFrequency());
	//cout << "   ������������ͼƬת��ʱ��:" << infer_time << endl;

	//string imageFilePath = "D:\\tf_train_my\\model\\20210722190514412.bmp";
	//Mat img = imread(imageFilePath);
	//string  output_csv = "D:\\tf_train_my\\output.xls";

	int count_pic = 0;

	vector<float> batch_res;

	//ѭ������ͼƬ����������Ԥ��

	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3, image_batch + i * _org_h * _org_w * 3);
		
		count_pic++;
		//cout << endl;
		//cout << "   ��+��+��+����ʼѭ����" << count_pic << "��ͼƬ" << endl;

		//cv::Mat img = sourceImage.clone();
		//cv::imwrite("1.jpg", img);

		//ѭ����ȡinput����blob��һ������ֻ��һ��
		for (auto item : _inputsInfo)
		{
			auto inpublob_begintime = cv::getTickCount();

			auto input_name = item.first;
			auto  framBlob = infer_request.GetBlob(input_name);
			 
			 //size
			_input_c = framBlob->getTensorDesc().getDims()[1];
			_input_h = framBlob->getTensorDesc().getDims()[2];
			_input_w = framBlob->getTensorDesc().getDims()[3];
			cout << "height:" << _input_h << endl;
			cout << "width:" << _input_w << endl;

			//Mat blob_image;
			//��ԭʼͼƬת��Ϊ����ͼƬ��С
			cv::resize(sourceImage, sourceImage, cv::Size(_input_w, _input_h));
			//cv::imwrite("2.jpg", sourceImage);

			sourceImage.convertTo(sourceImage, CV_32FC3, 1/255.0f);

			auto inpublob_endtime = cv::getTickCount();
			auto infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			//cout << "        ��������ͼƬresizeʱ��:" << infer_time << endl;

			inpublob_begintime = cv::getTickCount();

			//����input_blob����
			float* blob_data = static_cast<float*>(framBlob->buffer());

			std::vector<cv::Mat>channles(3);
			cv::split(sourceImage, channles);
			memcpy(blob_data + i * 3 * _input_h * _input_w, channles[0].data, _input_h * _input_w * sizeof(float));
			memcpy(blob_data + i * 3 * _input_h * _input_w + _input_h * _input_w, channles[1].data, _input_h * _input_w * sizeof(float));
			memcpy(blob_data + i * 3 * _input_h * _input_w + 2 * _input_h * _input_w, channles[2].data, _input_h * _input_w * sizeof(float));

			inpublob_endtime = cv::getTickCount();
			infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			//cout << "        ������������blobת��ʱ��:" << infer_time << endl;
		}

		//��ͼƬ��������
		auto infer_begintime = cv::getTickCount();
		try
		{
			//infer_request.Infer();
			infer_request.StartAsync();
			infer_request.Wait(InferRequest::WaitMode::RESULT_READY);

		}
		catch (const std::exception& ex)
		{
			//throw(ex);
		}

		auto infer_endtime = cv::getTickCount();
		infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		
		cout << "*����ʱ��ms:" << infer_time << endl;
				
		//���
		auto output_begintime= cv::getTickCount();

		//��ȡ������
		for (auto item : _outputinfo)
		{
			//1.��ȡ blob
			auto output_name = item.first;
			auto output_blob = infer_request.GetBlob(output_name);

			//�������cnts
            //1.��һ��ֱ�����ָ��
			parse_classify(output_blob, batch_res);
		}

		//stringstream ss;
		//ss << count_pic<<"_"<< max_index;
		//cv::imwrite(ss.str() +".jpg",img);
		//cout << "pred_res:" << max_index << endl;

		auto output_endtime = cv::getTickCount();
		infer_time = (to_string)((output_endtime - output_begintime) * 1000 / getTickFrequency());
		//cout << "����ʱ��ms:" << infer_time << endl;
	}
	//int num = 0;
	//for (auto item : batch_res)
	//{		
	//	cout << "��" << num << "��" << "score:" << item<< endl;
	//	num++;
	//}
	memcpy(prediction_batch, batch_res.data(), sizeof(float)* batch_res.size());
}


void ClassifyClass::process_frame2(unsigned char* image_batch, float* prediction_batch,
	int width, int height, int training_size, int batch_size)
{
	//��ȡԭʼ�������
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;

	//������������
	auto infer_request = _netWork.CreateInferRequest();

	//����ͼƬ��Ϣ��תΪmat����
	//std::vector<float> data;
	unsigned char* tempImage = image_batch;
	vector<Mat> org_pic;

	auto inputpic_begintime = cv::getTickCount();

	//����ÿ��batch�е�ͼƬ
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3);
		sourceImage.data = tempImage + i * _org_h * _org_w * 3;
		org_pic.push_back(sourceImage);
	}

	auto outputpic_endtime = cv::getTickCount();
	auto infer_time = (to_string)((outputpic_endtime - inputpic_begintime) * 1000 / getTickFrequency());
	//cout << "   ������������ͼƬת��ʱ��:" << infer_time << endl;

	int count_pic = 0;

	vector<float> batch_res;

	//ѭ������ͼƬ����������Ԥ��

	for (auto pic : org_pic)
	{
		count_pic++;

		Mat img = pic.clone();

		//ѭ����ȡinput����blob��һ������ֻ��һ��
		for (auto item : _inputsInfo)
		{
			auto inpublob_begintime = cv::getTickCount();

			auto input_name = item.first;
			auto  framBlob = infer_request.GetBlob(input_name);
			//cout << "dims_nums:" << endl;
			//cout << framBlob->getTensorDesc().getDims()[0] << endl;
			//cout << framBlob->getTensorDesc().getDims()[1] << endl;
			//cout << framBlob->getTensorDesc().getDims()[2] << endl;
			//cout << framBlob->getTensorDesc().getDims()[3] << endl;

			//
			_input_c = framBlob->getTensorDesc().getDims()[1];
			_input_h = framBlob->getTensorDesc().getDims()[3];
			_input_w = framBlob->getTensorDesc().getDims()[2];


			Mat blob_image;
			//��ԭʼͼƬת��Ϊ����ͼƬ��С
			resize(img, blob_image, Size(_input_w, _input_h));
			//ת��BGR��RGB
			cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

			auto inpublob_endtime = cv::getTickCount();
			auto infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			//cout << "        ��������ͼƬresizeʱ��:" << infer_time << endl;

			inpublob_begintime = cv::getTickCount();

			//����input_blob����
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
			//cout << "        ������������blobת��ʱ��:" << infer_time << endl;
		}

		//��ͼƬ��������
		auto infer_begintime = cv::getTickCount();
		try
		{
			//infer_request.Infer();
			infer_request.StartAsync();
			infer_request.Wait(InferRequest::WaitMode::RESULT_READY);
		}
		catch (const std::exception& ex)
		{
			//throw(ex);
		}

		auto infer_endtime = cv::getTickCount();
		infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());

		//cout << "   ��������������������������������������������������������������" << count_pic << "��ͼƬ���첽����ʱ��:" << infer_time << endl;

		////���ܹ�
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

		//���
		auto output_begintime = cv::getTickCount();

		//��ȡ������
		for (auto item : _outputinfo)
		{
			//1.��ȡ blob
			auto output_name = item.first;
			auto output_blob = infer_request.GetBlob(output_name);

			//�������cnts
			//1.��һ��ֱ�����ָ��
			parse_classify(output_blob, batch_res);
		}


		auto output_endtime = cv::getTickCount();
		infer_time = (to_string)((output_endtime - output_begintime) * 1000 / getTickFrequency());
		//cout << "   ����������" << count_pic << "��ͼƬ���������ʱ��:" << infer_time << endl;
		//cout << "����ʱ��:" << infer_time << endl;

	}
	//int num = 0;
	//for (auto item : batch_res)
	//{
	//	if (num % 2 == 0)
	//	{
	//		cout << "��" << num << "����Ƭ" << endl;

	//	}
	//	cout << "___" << item << endl;

	//}
	memcpy(prediction_batch, batch_res.data(), sizeof(float));

}

/// <summary>
/// ����output��������
/// </summary>
/// <param name="blob">����blob</param>
/// <param name="cnts">���uchar*</param>
/// <param name="width">ԭͼ���</param>
/// <param name="height">ԭͼ�߶�</param>
void ClassifyClass::parse_classify(const Blob::Ptr& blob, vector<float>& per_pic_res)
{
	//1.��ȡ�����dims
	auto outputDims = blob->getTensorDesc().getDims();

	//2.���ͼƬ�ߴ磬
	int output_class = outputDims[1];
	_output_class = output_class;


	//3.��ȡblob����
	LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
	float* output_buffer = blobMapped.as<float*>();

	for (size_t i = 0; i < output_class; i++)
	{
		per_pic_res.push_back(*(output_buffer + i));
	}	
}


