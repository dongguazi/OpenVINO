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
	//����һ��ת���ķ�ʽ�������ʱcv����matde ��ʽ�����resulting Blob pointer��Blob::Ptr��
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

double confidence_threshold = 0.99;  //���Ŷ���ֵ,���㷽���ǿ����Ŷȳ�����Ʒ�������Ŷ�
double nms_area_threshold = 0.99;  //nms��С�ص������ֵ

vector<string> labels = { "fake","cat","dog" };
/// <summary>
/// ��ò�ͬ����ê��
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
/// sigmoid����
/// </summary>
/// <param name="x"></param>
/// <returns></returns>
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}


int main(void)
{

	//0.���inferenceEgine��ǰ�汾��
	cout << "inferenceEgine:" << GetInferenceEngineVersion() << endl;


	//1.����coreģ��
	cout << "1.����coreģ�͡�����" << endl;

	Core ie;

	auto devicesVersions = ie.GetVersions(DEVICE);
	for (auto iter = devicesVersions.rbegin(); iter != devicesVersions.rend(); iter++)
	{
		cout << "    �豸���ƣ�" << iter->first << "���豸�汾�ţ�" << iter->second << endl;
	}
	auto devices = ie.GetAvailableDevices();
	for (auto iter = devices.begin(); iter != devices.end(); iter++)
	{
		cout << "    �����豸����1��" << iter->c_str() << endl;

	}
	//for (int i = 0; i < devices.size(); i++)
	//{
	//	cout << "�����豸����2��" << devices[i] << endl;
	//}


	//2.��ȡIRģ���ļ�
	cout << "2.��ȡIRģ�͡�����" << endl;
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

	cout << "    ����ʱ�䣺" << (to_string)(end1 - begin1) << "ms" << endl;
	cout << "    ��ȡIRģ����ɣ�" << endl;

	//3.�����������
	cout << "3.�����������������" << endl;
	string imageInputName, imInfoInputName;
	//InputInfo::Ptr input_data = nullptr; 

	string imageOutputName, imInfoOutputName;
	//InputInfo::Ptr output_data = nullptr;

	//��������
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
	//�������
	OutputsDataMap outputsInfo = network.getOutputsInfo();
	for (auto& item : outputsInfo)
	{
		imageOutputName = item.first;
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
		//output_data->setLayout(Layout::NC);
	}
	cout << "    �������������ɣ�" << endl;


	//4.����ģ��
	cout << "4.����ģ�͡�����" << endl;
	cout << "    ��ʼ����ģ�ͣ�" << endl;

	clock_t loadtimebegin;
	clock_t loadtimeend;
	loadtimebegin = clock();
	auto executable_network = ie.LoadNetwork(network, DEVICE);
	loadtimeend = clock();
	cout << "    ����ʱ�䣺" << (to_string)(loadtimeend - loadtimebegin) << "ms" << endl;

	cout << "    ����ģ����ɣ�" << endl;


	//5.��������
	cout << "5.�����������󡣡���" << endl;
	auto infer_request = executable_network.CreateInferRequest();
	cout << "    ����������ɣ�" << endl;


	//6.�������ݴ���
	//��opencv�е�mat����ת��ΪinferenceEngineBlob���󣬻��ͼ������
	cout << "6.׼���������ݡ�����" << endl;
	Mat img = cv::imread(imageFile);
	if (img.empty())
	{
		cout << "      ��ЧͼƬ����" << endl;
	}
	Mat img_copy = img.clone();
	cout << "       ԭͼͨ������" << img_copy.channels() << endl;

	resize(img_copy, img_copy, Size(640, 640));


	auto width = img_copy.cols;
	auto height = img_copy.rows;

	//imshow("����ͼ��", img);  

	for (auto& item : inputsInfo)
	{
		auto input_name = item.first;
		auto  input = infer_request.GetBlob(input_name);

		cout << "     ����blob�ĸ���" << input->byteSize() << endl;;
		//cout << "����blob�ĸ���" << input->buffer() << endl;;


		//����1������frameToBlob��ֱ���޸�infer_request��buffer����
		//frameToBlob(img_copy, infer_request, input_name);


		//����2������wrapMat2Blob�����������ٴ���input_name��
		auto data_blob = wrapMat2Blob(img_copy);
		infer_request.SetBlob(input_name, data_blob);


		//����3���Լ�д��������matת��Ϊblob::Ptr

#pragma region MyRegion
//���ģ�͵�CHW
		//size_t num_channel = input->getTensorDesc().getDims()[1];
		//size_t h = input->getTensorDesc().getDims()[2];
		//size_t w = input->getTensorDesc().getDims()[3];

		//size_t image_size = h * w;
		//Mat blob_image;
		//// �ܽ᣺ģ����˳��(h,w)��channelΪRGB��opencv��˳��(rows,cols)��(h,w),channelΪBGR,
		//// �������ͼ��ת��Ϊģ�Ϳ��Խ��ܵĳߴ�resize
		//resize(img_copy, blob_image, Size(w, h));

		////opencv��ȡ��ͼ��channel��BGR����Ҫת����ģ�Ϳ��õ�RGB
		//cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

		////��RGBͼ��ת��Ϊ������õ�blob::Ptr����������д�뵽������
		////data ֱ��ָ����buffer���ڴ��ַ���޸�data����ֱ���޸�infer��buffer�ڵ�ֵ(blob::Ptr)
		//cout << "ԭͼͨ������" << blob_image.channels() << endl;

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

	cout << "    ����������ɣ�" << endl;


	//7.����
	cout << "7.��������" << endl;
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

	//infer_time << "����ʱ�䣺" << (double)(end - begin) << "ms";
	//putText(img, "time:" + infer_time + "ms", Point2f(0, 12), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0, 255, 0));

	cout << "    ������ɣ�" << endl;


	//8.���
	cout << "8.���������" << endl;

	//��ȡ������
	vector<Rect> origin_rect;
	vector<float> origin_rect_cof;

	begintime = cv::getTickCount();

	//auto detections = infer_request.GetBlob(imageOutputName);
	for (auto item : outputsInfo)
	{
		//1.��ȡ blob
		auto output_name = item.first;
		auto output = infer_request.GetBlob(output_name);

		//2.��ȡ�����dims
		auto outputDims = output->getTensorDesc().getDims();
		cout << output_name << "    ���ά��ά��:" << outputDims.size() << endl;
		//��������ĸ�ʽΪ��n��layers��h,w,out
		// out[dim=4]=85,Ϊoffset_x��offset_y,scale_h,scale_w,obj_confid,probs=80
		//��һ������ߴ�417��1,3,40,40,85
		//�ڶ�������ߴ�437��1��3��20��20��85
		//����������ߴ�output��1��3��80��80��85 	

		//���ͼƬ�ߴ磬������Ӧ��anchors��ÿ�����ĵ�������anchors
		int net_grid = outputDims[2];
		vector<int> anchors = get_anchors(net_grid);

		//��ȡ��������ά��
		int net_output_size = outputDims[4];
		size_t anchor_n = 3;

		//3.��ȡblob����ָ��
	/*	auto const memLocker = output->cbuffer();
		const float* output_buffer = memLocker.as <const float*>();*/

		LockedMemory<const void> blobMapped = as<MemoryBlob>(output)->rmap();
		const float* output_buffer = blobMapped.as<float*>();
		//4.�������Ŷ�cofid��rect
		for (int n = 0; n < anchor_n; ++n)
			for (int i = 0; i < net_grid; ++i)
				for (int j = 0; j < net_grid; ++j)
				{
					int object_index = n * net_grid * net_grid * net_output_size + i * net_grid * net_output_size + j * net_output_size;

					/*			for (int i = 0; i < 15; i++)
								{
									cout << "   �����" <<to_string(i) << "==" << output_buffer[object_index + i]<<endl;
								}*/
								//ÿ��ͼ
					double box_prob = output_buffer[object_index + 4];
					box_prob = sigmoid(box_prob);
					//��ǰ���ŶȲ������������Ŷ�
					if (box_prob < confidence_threshold)
					{
						continue;
					}
					//�˴����Ϊ���ĵ����꣬Ҫ��Ϊ��������
					double x = output_buffer[object_index + 0];
					double y = output_buffer[object_index + 1];
					double w = output_buffer[object_index + 2];
					double h = output_buffer[object_index + 3];

					double max_prob = 0;
					int idx = 0;
					for (int t = 5; t < net_output_size; ++t)
					{
						//�ҵ�80������е÷���ߵ����
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


	//9.�������ս��
	vector<int> final_id;
	dnn::NMSBoxes(origin_rect, origin_rect_cof, confidence_threshold, nms_area_threshold, final_id);

	for (int i = 0; i < final_id.size(); ++i)
	{
		Rect resize_rect = origin_rect[final_id[i]];
		double cof = origin_rect_cof[final_id[i]];
		cout << "     �÷֣�" << to_string(cof) << endl;

		putText(img_copy, to_string(cof), Point2f(resize_rect.x, resize_rect.y - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
		rectangle(img_copy, resize_rect, Scalar(255, 0, 0), 1, LINE_8);

	}
	//10.��������ʱ��
	endtime = cv::getTickCount();
	auto infer_time = (to_string)((endtime - begintime) * 1000 / getTickFrequency());
	putText(img_copy, "time:" + infer_time + "ms", Point2f(0, 12), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0, 255, 0));


	cout << "      �����ɣ�" << endl;

	cout << "9.ȫ����ɣ�������ͼ��" << endl;

	imshow("�����", img_copy);
	waitKey(0);
	destroyAllWindows();
	return 0;
}