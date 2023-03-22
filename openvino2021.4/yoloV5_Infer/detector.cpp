#include"detector.h";

Yolov5Dectector::Yolov5Dectector(){};

Yolov5Dectector::~Yolov5Dectector() {};

/// <summary>
/// 初始化网络
/// </summary>
/// <param name="device">设备="CPU""GPU"</param>
/// <param name="xml_path">xml文件路径</param>
bool Yolov5Dectector::InitializeDetector(string device,string xml_path)
{
	_xmlpath = xml_path;
	_device = device;

	//建立IEcore
	Core ie;
	CNNNetwork network;
	try
	{
		network = ie.ReadNetwork(_xmlpath);
		//network = ie.ReadNetwork(ONNX_file);
		cout << "*****1.读取模型文件"<< endl;
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
	}
	catch (const std::exception& ex)
	{
		std::cout << "" << ex.what() << endl;
		cout << "      模型加载失败！" << device << endl;

	}

	cout << "      模型加载完成！" << device << endl;

	return true;
}

bool Yolov5Dectector::uninit() {
	return true;
}

/// <summary>
/// sigmoid函数
/// </summary>
/// <param name="x"></param>
/// <returns></returns>
float Yolov5Dectector::sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

/// <summary>
/// 获得不同规格的锚框
/// </summary>
/// <param name="net_grid"></param>
/// <returns></returns>
vector<int> Yolov5Dectector::get_anchors(int net_grid)
{
	vector<int> anchors(6);
	//int achors80[6] = { 10,13, 16,30, 33,23 };
	//int achors40[6] = { 30,61, 62,45, 59,119 };
	//int achors20[6] = { 116,90, 156,198, 373,326 };
	int achors80[6] = { 10,13, 16,30, 33,23 };
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
/// 处理输入图像
/// </summary>
/// <param name="inframe"></param>
/// <param name="out_detectod_result"></param>
/// <returns></returns>
void Yolov5Dectector::process_frame(unsigned char* image_batch, float* x1_ptr,
	float* y1_ptr, float* x2_ptr, float* y2_ptr, float* prob_ptr, int* class_ptr, 
	int* num_boxes_ptr, int buffer_size, int width, int height, 
	int batch_size, float conf_thresh, float nms_thres)
{
	//获取原始输入参数
	_buffer_size = buffer_size;
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;
	_cof_threshold = conf_thresh;
	_nms_IOU = nms_thres;

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
	//循环所有图片，进行推理预测
	for (auto pic : org_pic)
	{
		count_pic++;
		cout  << endl;
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

			//计算原始图片和输入图片的比例
			_org_h_scale = float(_org_h) / _input_h;
			_org_w_scale = float(_org_w) / _input_w;

			Mat blob_image;
			//将原始图片转换为输入图片大小
			resize(img, blob_image, Size(_input_w, _input_h));
			//转换BGR到RGB
			cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

			auto inpublob_endtime = cv::getTickCount();
			auto infer_time = (to_string)((inpublob_endtime - inpublob_begintime) * 1000 / getTickFrequency());
			cout << "        ————图片resize时间:" << infer_time << endl;


			inpublob_begintime = cv::getTickCount();
			//用属兔图像更新blob数据
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

		//infer_request.Infer();
		infer_request.StartAsync();
		infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);

		auto infer_endtime = cv::getTickCount();
	    infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		cout <<"   ————第"<< count_pic <<"张图片的异步推理时间:" << infer_time << endl;

		//输出
		vector<Rect> origin_rect;
		vector<float> origin_rect_cof;
		vector<int> origin_class_pred;


		auto parse_begintime = cv::getTickCount();

		//vector<YoloV4Result> result;
		//获取输出结果
		for (auto item : _outputinfo)
		{
			//1.获取 blob
			auto output_name = item.first;
			auto output_blob = infer_request.GetBlob(output_name);

			parse_yolov4(output_blob, _cof_threshold, _input_h, _input_w,_class_nums, origin_rect, origin_rect_cof, origin_class_pred);
		}

		auto parse_endtime = cv::getTickCount();
	    infer_time = (to_string)((parse_endtime - parse_begintime) * 1000 / getTickFrequency());
		cout << "        ————解析输出时间:" << infer_time << endl;
		//打印数据
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
		cout << "        ————nms转换时间:" << infer_time << endl;

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
		cout << "        ————结果输出时间:" << infer_time << endl;
	}	
}

/// <summary>
/// 推理
/// </summary>
/// <param name="blob"></param>
/// <param name="cof_threshold"></param>
/// <param name="input_h"></param>
/// <param name="input_w"></param>
/// <param name="class_nums"></param>
/// <param name="o_rect"></param>
/// <param name="o_rect_cof"></param>
/// <param name="o_class_pred"></param>
void Yolov5Dectector::parse_yolov4(const Blob::Ptr& blob, float cof_threshold,
	int input_h, int input_w,int class_nums,
	vector<Rect>& o_rect, vector<float>& o_rect_cof,vector<int>& o_class_pred)
{   
	//1.获取输出的dims
	auto outputDims = blob->getTensorDesc().getDims();
	
	//输出格式【n,c,w,h,85】
    //85=[x,y,w,h,prob,class]
	// 
	//锚框个数指定为3
	int anchor_n = 3;
	//2.获得图片尺寸，
	int net_grid_w = outputDims[2];
	int net_grid_h = outputDims[3];
	int output_size = outputDims[4];
	_class_nums = output_size - 5;

	//3.获取锚框
	vector<int> anchors = get_anchors(net_grid_w);
	
	//4.获取blob数据
	LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
	float* output_buffer = blobMapped.as<float*>();


	//5.计算置信度cofid和rect
	_stride_w = float(input_w) / net_grid_w;
	_stride_h = float(input_h) / net_grid_h;


	for (int n = 0; n < anchor_n; n++)
	{
		//判定通道数据的起始点
	    //85=[x,y,w,h,prob,class]

		//将anchors的高宽按照input的分辨率转换到grid上
		   //所有计算都是基于grid，在grid上预测好后，再转换到input上
		const float anchor_w = anchors[2 * n]/ _stride_w;
		const float anchor_h = anchors[2 * n + 1]/ _stride_h;

		for (int i = 0; i < net_grid_h; i++)
		{
			for (int j = 0; j < net_grid_w; j++)
			{
				//取某个像素点的三个锚框，指针每次指针跳动11次，
					//第一个锚点是p，第二个锚点是p+11，第三个锚点是p+11+11
					//float* row = output_buffer + n * (class_nums + 5);
					//conf数据起始点
				int position = n * net_grid_w * net_grid_h * output_size + i * net_grid_w * output_size + j * output_size;
				int x_index = position +0;
				int y_index = position + 1;
				int w_index = position + 2;
				int h_index = position + 3;
				int conf_index = position + 4;

				//计算conf，阈值过虑
				double box_cof = sigmoid(output_buffer[conf_index]);

				if (box_cof < _cof_threshold)
				{
					continue;
				}

				//找到分类中最大元素的位置
				//float max_prob = -1;
				float max_prob = -10;
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
					auto prob = output_buffer[position+5+ t];
					prob = sigmoid(prob);
					if (prob > max_prob)
					{
						max_prob = prob;
						idx = t;
					}
				}
				//max_prob = sigmoid(max_prob);

				//计算cof，等于max_prob * box_prob
				float cof = max_prob * box_cof;
				//对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
				if (cof < _cof_threshold)
				{
					continue;
				}
				//head的decode过程
				//解析cx,cy,pw,ph，
					//pw=anchors[n * 2]*grid_w/input_w，为当前anchors的宽度
					//ph= anchors[n * 2+1]*grid_h/input_h，为当前anchors的高度
					//cx=input_w/grid_w*i=512/grid_w*i，为当前中心点在方格宽度偏移的个数j
					//cy=input_h/grid_h*j=512/grid_h*j，为当前中心点在方格高度偏移的个数i
				//解析最终坐标
					//bx=sigmoid(tx)+cx
					//by=sigmoid(ty)+cy
					//bw=pw*exp(tw)
					//bh=ph*exp(th)
					// 
				//计算类别
				int class_label = idx;

				//float x = (sigmoid(center_x) * 2 - 0.5 + j) * _stride_w;
				//float y = (sigmoid(center_y) * 2 - 0.5 + i) * _stride_h;
				//float w = pow(sigmoid(output_buffer[w_index]) * 2, 2) * anchors[n * 2];
				//float h = pow(sigmoid(output_buffer[h_index]) * 2, 2) * anchors[n * 2 + 1];
				
				float x = output_buffer[x_index];
				float y = output_buffer[y_index];
				float w = output_buffer[w_index];
				float h = output_buffer[h_index];



				 x = (sigmoid(x)*2-0.5+j)* _stride_w;
				 y = (sigmoid(y)*2-0.5+i) * _stride_h;
				 w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
				 h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];
				//float w = exp(sigmoid(output_buffer[w_index])*2)* anchor_w * _stride_w;
				//float h = exp(sigmoid(output_buffer[h_index]) * 2) * anchor_h * _stride_h;

				//x = x + j;
				//y = y + i;
				//w = exp(w) * anchor_w;
				//h = exp(h) * anchor_h;
				//计算rect
				float r_x1 = (x - w / 2);
				float r_y1 = (y - h / 2);
				float r_x2 = (x + w / 2);
				float r_y2 = (y + h / 2);

				//YoloV4Result res = YoloV4Result();;
				//res.x1 = r_x1;
				//res.y1 = r_y1;
				//res.x2 = r_x2;
				//res.y2 = r_y2;
				//res.class_num = class_label;
				//res.prob = cof;

				//yolov4_result.push_back(res);

				Rect rect = Rect(round(r_x1), round(r_y1), round(w), round(h));
				//Rect rect = Rect(*new Point2i(r_x1, r_y1), *new Point2i(r_x2, r_y2));
				o_rect.push_back(rect);
				o_rect_cof.push_back(cof);
				o_class_pred.push_back(class_label);

			}
		}
	}
}
