#include"detector.h";
#include"LicenseChecker.h"

Yolov4Dectector::Yolov4Dectector(){};

Yolov4Dectector::~Yolov4Dectector() {};

/// <summary>
/// ��ʼ������
/// </summary>
/// <param name="device">�豸="CPU""GPU"</param>
/// <param name="xml_path">xml�ļ�·��</param>
bool Yolov4Dectector::InitializeDetector(string device,string xml_path)
{
	_xmlpath = xml_path;
	_device = device;
	cout << "*****0.��ʼ����ʼִ��" << endl;


	//����IEcore
	Core ie;
	CNNNetwork network;
	try
	{

		network = ie.ReadNetwork(_xmlpath);

		cout << "*****1.��ȡģ���ļ�"<< endl;
	}
	catch (const std::exception& ex)
	{
		std::cout << "" << ex.what() << endl;
	}

	network.setBatchSize(1);

	auto devices = ie.GetAvailableDevices();
	cout << "*****2.���ص�ǰ�����豸Ϊ��" << device << endl;
	for (auto iter = devices.begin(); iter != devices.end(); iter++)
	{
		cout << "    �����豸����1��" << iter->c_str() << endl;
	}

	//��������
	_inputsInfo= network.getInputsInfo();

	for (auto& item : _inputsInfo)
	{
		_inputname = item.first;
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
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
	cout << "*****3.��ȴ���ģ�ͼ�����........" << device << endl;

	try
	{
		auto load_begintime = cv::getTickCount();
		_netWork = ie.LoadNetwork(network, _device);
		auto load_endtime = cv::getTickCount();
		auto infer_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
		cout << "      *****ģ�ͼ���ʱ��ʱ��ms:" << infer_time << endl; 
		cout << "      ģ�ͼ�����ɣ�" << device << endl;
	}
	catch (const std::exception& ex)
	{
		std::cout << "" << ex.what() << endl;
		cout << "      ģ�ͼ���ʧ�ܣ�" << device << endl;
		return false;
	}

	return true;
}

bool Yolov4Dectector::uninit() {
	return true;
}

/// <summary>
/// sigmoid����
/// </summary>
/// <param name="x"></param>
/// <returns></returns>
double Yolov4Dectector::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

/// <summary>
/// ��ò�ͬ����ê��
/// </summary>
/// <param name="net_grid"></param>
/// <returns></returns>
vector<int> Yolov4Dectector::get_anchors(int net_grid)
{
	vector<int> anchors(6);
	//int achors80[6] = { 10,13, 16,30, 33,23 };
	//int achors40[6] = { 30,61, 62,45, 59,119 };
	//int achors20[6] = { 116,90, 156,198, 373,326 };
	int achors80[6] = { 2, 16,  19, 36,  40, 28 };
	int achors40[6] = { 36, 75,  76, 55,  72, 146 };
	int achors20[6] = { 142, 110,  192, 243,  459, 401 };
	if (net_grid == 64)
	{
		anchors.insert(anchors.begin(), achors80, achors80 + 6);

	}
	else if (net_grid == 32)
	{
		anchors.insert(anchors.begin(), achors40, achors40 + 6);

	}
	else if (net_grid == 16)
	{
		anchors.insert(anchors.begin(), achors20, achors20 + 6);

	}
	return anchors;
}

/// <summary>
/// ��������ͼ��
/// </summary>
/// <param name="inframe"></param>
/// <param name="out_detectod_result"></param>
/// <returns></returns>
void Yolov4Dectector::process_frame(unsigned char* image_batch, float* x1_ptr,
	float* y1_ptr, float* x2_ptr, float* y2_ptr, float* prob_ptr, int* class_ptr, 
	int* num_boxes_ptr, int buffer_size, int width, int height, 
	int batch_size, float conf_thresh, float nms_thres)
{
	//��ȡԭʼ�������
	_buffer_size = buffer_size;
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;
	_cof_threshold = conf_thresh;
	_nms_IOU = nms_thres;

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
	cout << "   ������������ͼƬת��ʱ��:" << infer_time << endl;


	//string imageFilePath = "D:\\tf_train_my\\model\\20210722190514412.bmp";
	//Mat img = imread(imageFilePath);
	//string  output_csv = "D:\\tf_train_my\\output.xls";

	int count_pic = 0;
	int num = 0;
	//ѭ������ͼƬ����������Ԥ��
	for (auto pic : org_pic)
	{
		count_pic++;
		cout  << endl;
		cout << "   ��+��+��+����ʼѭ����" << count_pic << "��ͼƬ" << endl;

		Mat img = pic.clone();

		//ѭ����ȡinput����blob��һ������ֻ��һ��
		for (auto item : _inputsInfo)
		{
			auto inpublob_begintime = cv::getTickCount();

			auto input_name = item.first;
			auto  framBlob = infer_request.GetBlob(input_name);

			_input_c = framBlob->getTensorDesc().getDims()[1];
			_input_h = framBlob->getTensorDesc().getDims()[2];
			_input_w = framBlob->getTensorDesc().getDims()[3];

			//����ԭʼͼƬ������ͼƬ�ı���
			_org_h_scale = float(_org_h) / _input_h;
			_org_w_scale = float(_org_w) / _input_w;

			Mat blob_image;
			//��ԭʼͼƬת��Ϊ����ͼƬ��С
			resize(img, blob_image, Size(_input_w, _input_h));
			//ת��BGR��RGB
			cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

			auto inpublob_endtime = cv::getTickCount();
			auto infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			cout << "        ��������ͼƬresizeʱ��:" << infer_time << endl;


			inpublob_begintime = cv::getTickCount();
			//������ͼ�����blob����
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
			cout << "        ������������blobת��ʱ��:" << infer_time << endl;

		}

		//��ͼƬ��������
		auto infer_begintime = cv::getTickCount();

		//infer_request.Infer();
		infer_request.StartAsync();
		infer_request.Wait(InferRequest::WaitMode::RESULT_READY);

		auto infer_endtime = cv::getTickCount();
	    infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		cout <<"   ����������"<< count_pic <<"��ͼƬ���첽����ʱ��:" << infer_time << endl;

		//���
		vector<Rect> origin_rect;
		vector<float> origin_rect_cof;
		vector<int> origin_class_pred;


		auto parse_begintime = cv::getTickCount();

		//vector<YoloV4Result> result;
		//��ȡ������
		for (auto item : _outputinfo)
		{
			//1.��ȡ blob
			auto output_name = item.first;
			auto output_blob = infer_request.GetBlob(output_name);

			parse_yolov4(output_blob, _cof_threshold, _input_h, _input_w,_class_nums, origin_rect, origin_rect_cof, origin_class_pred);
		}

		auto parse_endtime = cv::getTickCount();
	    infer_time = (to_string)((parse_endtime - parse_begintime) * 1000 / getTickFrequency());
		cout << "        ���������������ʱ��:" << infer_time << endl;
		//��ӡ����
		//ofstream of = ofstream(output_csv);

		//for (auto item: yolov4_result)
		//{
		//	float x1 = item.x1;
		//	float y1 = item.y1;
		//	float x2 = item.x2;
		//	float y2 = item.y2;
		//	float calss = item.class_num;
		//	float cof = item.prob;
		//	of << x1 << "\t";
		//	of << y1 << "\t";
		//	of << x2 << "\t";
		//	of << y2 << "\t";
		//	of << calss << "\t";
		//	of << cof << endl;
		//}
		//of.close();

		auto nms_begintime = cv::getTickCount();

		vector<int> final_id;
		dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_IOU, final_id);
		
		auto nms_endtime = cv::getTickCount();
		infer_time = (to_string)((nms_endtime - nms_begintime) * 1000 / getTickFrequency());
		cout << "        ��������nmsת��ʱ��:" << infer_time << endl;

		auto out_begintime = cv::getTickCount();

		int nums_boxs = final_id.size();
		int output_length = min(nums_boxs,buffer_size);

		memcpy(num_boxes_ptr, &output_length, sizeof(int));
		num_boxes_ptr++;


		for (int i = 0; i < output_length; i++)
		{
			float x1 = origin_rect[final_id[i]].x *_org_w_scale;
			float y1= origin_rect[final_id[i]].y * _org_h_scale;
			float x2 = (origin_rect[final_id[i]].x + origin_rect[final_id[i]].width) * _org_w_scale;
			float y2 = (origin_rect[final_id[i]].y + origin_rect[final_id[i]].height) * _org_h_scale;
			float prob_cof = origin_rect_cof[final_id[i]];
			int class_pred = origin_class_pred[final_id[i]];

			memcpy(x1_ptr + num, &x1, sizeof(float));
			memcpy(y1_ptr + num, &y1, sizeof(float));
			memcpy(x2_ptr + num, &x2, sizeof(float));
			memcpy(y2_ptr + num, &y2, sizeof(float));
			memcpy(prob_ptr + num, &prob_cof, sizeof(float));
			memcpy(class_ptr + num, &class_pred, sizeof(int));
			num++;
		}

		auto out_endtime = cv::getTickCount();
         infer_time = (to_string)((out_endtime - out_begintime) * 1000 / getTickFrequency());
		cout << "        ��������������ʱ��:" << infer_time << endl;
	}	
}

/// <summary>
/// ����
/// </summary>
/// <param name="blob"></param>
/// <param name="cof_threshold"></param>
/// <param name="input_h"></param>
/// <param name="input_w"></param>
/// <param name="class_nums"></param>
/// <param name="o_rect"></param>
/// <param name="o_rect_cof"></param>
/// <param name="o_class_pred"></param>
void Yolov4Dectector::parse_yolov4(const Blob::Ptr& blob, float cof_threshold,
	int input_h, int input_w,int class_nums,
	vector<Rect>& o_rect, vector<float>& o_rect_cof,vector<int>& o_class_pred)
{   

	//char* code = CheckLicense();
	//cout << code << endl;
	//if (code == NULL)
	//{
	//	throw exception("License Error");
	//}
	//std::string name = string(code);

	//cout << "name" << name << endl;
	//size_t postion = name.find("VxDeepVino");
	////size_t postion = name.find("VxDeepVino");


	//if (postion == string::npos)
	//	throw exception("License Error");

	//1.��ȡ�����dims
	auto outputDims = blob->getTensorDesc().getDims();
	
	//ê�����ָ��Ϊ3
	int anchor_n = 3;
	//2.���ͼƬ�ߴ磬
	int output_size = outputDims[1];
	int net_grid_w = outputDims[2];
	int net_grid_h = outputDims[3];
	_class_nums = output_size / anchor_n - 5;

	//�������Ϊ11��4��anchors��+1(class)+6��pred��
	int output_size_per = output_size / anchor_n;
    
	//3.��ȡê��
	vector<int> anchors = get_anchors(net_grid_w);
	
	//4.��ȡblob����
	LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
	float* output_buffer = blobMapped.as<float*>();


	//5.�������Ŷ�cofid��rect
	int anchors_position = 0;
	int count = 0;
	_stride_w = float(input_w) / net_grid_w;
	_stride_h = float(input_h) / net_grid_h;
	
	//head��ʽ[bs,boxnums,nx,ny,[x,y,w,h,cof,cls1,cls2...cls80]]
	for (int n = 0; n < anchor_n; n++)
	{
		//�ж�ͨ�����ݵ���ʼ��
		anchors_position = n * output_size_per * net_grid_w * net_grid_h;
		count = 0;

		//��anchors�ĸ߿���input�ķֱ���ת����grid��
		   //���м��㶼�ǻ���grid����grid��Ԥ��ú���ת����input��
		const float anchor_w = anchors[2 * n]/ _stride_w;
		const float anchor_h = anchors[2 * n + 1]/ _stride_h;

		for (int i = 0; i < net_grid_h; i++)
		{
			for (int j = 0; j < net_grid_w; j++)
			{
				//ȡĳ�����ص������ê��ָ��ÿ��ָ������11�Σ�
					//��һ��ê����p���ڶ���ê����p+11��������ê����p+11+11
					//float* row = output_buffer + n * (class_nums + 5);
					//conf������ʼ��
				int x_index = anchors_position + 0 * net_grid_w * net_grid_h + count;
				int y_index = anchors_position + 1 * net_grid_w * net_grid_h + count;
				int w_index = anchors_position + 2 * net_grid_w * net_grid_h + count;
				int h_index = anchors_position + 3 * net_grid_w * net_grid_h + count;
				int conf_index = anchors_position + 4 * net_grid_w * net_grid_h + count;

				//����conf����ֵ����
				double box_prob = sigmoid(output_buffer[conf_index]);

				//�ҵ����������Ԫ�ص�λ��
				float max_prob = 0;
				float class_conf = -1000;
				int idx = 0;
		/*		for (int t = 0; t < _class_nums; t++)
				{
					auto temp = output_buffer[anchors_position + (t+5) * net_grid_w * net_grid_h + count];
					temp = sigmoid(temp);
					if (temp > max_prob)
					{
						max_prob = temp;
						idx = t;
					}
				}*/
				for (int t = 0; t < _class_nums; t++)
				{
					auto temp = output_buffer[anchors_position + (t + 5) * net_grid_w * net_grid_h + count];
		
					if (temp > class_conf)
					{
						class_conf = temp;
						idx = t;
					}
				}
				class_conf = sigmoid(class_conf);

				//����cof������max_prob * box_prob
				float cof = class_conf * box_prob;
				//���ڱ߿����Ŷ�С����ֵ�ı߿�,������������ֵ,�����м�����ټ�����
				if (cof < _cof_threshold)
				{
					count++;
					continue;
				}
				//head��decode����
				//����cx,cy,pw,ph��
					//pw=anchors[n * 2]*grid_w/input_w��Ϊ��ǰanchors�Ŀ��
					//ph= anchors[n * 2+1]*grid_h/input_h��Ϊ��ǰanchors�ĸ߶�
					//cx=input_w/grid_w*i=512/grid_w*i��Ϊ��ǰ���ĵ��ڷ�����ƫ�Ƶĸ���j
					//cy=input_h/grid_h*j=512/grid_h*j��Ϊ��ǰ���ĵ��ڷ���߶�ƫ�Ƶĸ���i
				//������������
					//bx=sigmoid(tx)+cx
					//by=sigmoid(ty)+cy
					//bw=pw*exp(tw)
					//bh=ph*exp(th)

				float center_x = output_buffer[x_index];
				float center_y = output_buffer[y_index];
				float center_w = output_buffer[w_index];
				float center_h = output_buffer[h_index];
				count++;

				//�������
				int class_label = idx;

				//float x = (sigmoid(center_x) * 2 - 0.5 + j) * _stride_w;
				//float y = (sigmoid(center_y) * 2 - 0.5 + i) * _stride_h;
				//float w = pow(sigmoid(center_w) * 2, 2) * anchors[n * 2];
				//float h = pow(sigmoid(center_h) * 2, 2) * anchors[n * 2 + 1];
				
				float x = sigmoid(center_x);
				float y = sigmoid(center_y);
				float w = (center_w);
				float h = (center_h);

				x = x + j;
				y = y + i;
				w = exp(w) * anchor_w;
				h = exp(h) * anchor_h;
				//����rect
				float r_x1 = (x - w / 2)* _stride_w;
				float r_y1 = (y - h / 2)*_stride_h;
				float r_x2 = (x + w / 2)* _stride_w;
				float r_y2 = (y + h / 2) * _stride_h;

				//YoloV4Result res = YoloV4Result();;
				//res.x1 = r_x1;
				//res.y1 = r_y1;
				//res.x2 = r_x2;
				//res.y2 = r_y2;
				//res.class_num = class_label;
				//res.prob = cof;

				//yolov4_result.push_back(res);

				Rect rect = Rect(round(r_x1), round(r_y1), round(r_x2 - r_x1), round(r_y2 - r_y1));
				//Rect rect = Rect(*new Point2i(r_x1, r_y1), *new Point2i(r_x2, r_y2));
				o_rect.push_back(rect);
				o_rect_cof.push_back(cof);
				o_class_pred.push_back(class_label);

			}
		}
	}
}
