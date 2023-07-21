#include"detector.h";
#include"LicenseChecker.h"

Yolov4Dectector::Yolov4Dectector(){};

Yolov4Dectector::~Yolov4Dectector() {};

/// <summary>
/// 初始化网络
/// </summary>
/// <param name="device">设备="CPU""GPU"</param>
/// <param name="xml_path">xml文件路径</param>
bool Yolov4Dectector::InitializeDetector(string device,string _onnx_path)
{
	_onnxpath = _onnx_path;
	_device = device;
	//cout << "*****0.初始化开始执行" << endl;

	// -------- Step 2. Read a model --------
	try
	{
		auto load_begintime = cv::getTickCount();

		model = core.read_model(_onnx_path);

		_model_input_nums = model->inputs().size();
		cout << "input size:" << _model_input_nums << endl;
		for (size_t i = 0; i < _model_input_nums; i++)
		{
			cout << "input_name_" << i << ":" << model->inputs()[i] << endl;
			cout << "input_type_" << i << ":" << model->input().get_element_type() << endl;
			cout << "input_shape_" << i << ":" << model->input().get_shape() << endl;
		}

	    _model_output_nums = model->outputs().size();
		cout << "output size:" << _model_output_nums << endl;
		for (size_t i = 0; i < _model_output_nums; i++)
		{
			cout << "output_name_" << i << ":" << model->outputs()[i] << endl;
			cout << "output_type_"<<i <<":" << model->get_output_element_type(i) << endl;
			cout << "output_shape_" << i << ":" << model->get_output_shape(i) << endl;
		}



		auto load_endtime = cv::getTickCount();
		auto infer_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
		cout << "*模型加载成功！" << device << "  *加载时间ms:" << infer_time << endl;
	}
	catch (const std::exception& ex)
	{
		cout << "*模型加载失败！" << device << endl;
		return false;
	}

	// -------- Step 3. Configure preprocessing --------
	ov::preprocess::PrePostProcessor ppp(model);
	// 1) input() with no args assumes a model has a single input
	ov::preprocess::InputInfo& input_info = ppp.input();
	// 2) Set input tensor information:
	// - precision of tensor is supposed to be 'u8',some model need f32 or f16
	// - layout of data is 'NHWC'
	const ov::Layout tensor_layout{ "NCHW" };
	for (size_t i = 0; i < _model_input_nums; i++)
	{
		ppp.input(i).tensor().set_element_type(ov::element::f32).set_layout(tensor_layout);
		ppp.input(i).preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
		//ppp.input(i).preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
	}

	// 3) Here we suppose model has 'NCHW' layout for input
	input_info.model().set_layout("NCHW");

	// 4) output() with no args assumes a model has a single result
	// - precision of tensor is supposed to be 'f32'
	//ov::preprocess::OutputInfo& output_info = ppp.output();
	for (size_t i = 0; i < _model_output_nums; i++)
	{
		ppp.output(i).tensor().set_element_type(ov::element::f32);
	}

	// 5) Once the build() method is called, the pre(post)processing steps
	// for layout and precision conversions are inserted automatically
	model = ppp.build();

	//设置模型的batch size
	ov::set_batch(model, 1);

	// -------- Step 4. Loading model to the device --------
	auto load_begintime = cv::getTickCount();
	compiled_model = core.compile_model(model, _device);
	auto load_endtime = cv::getTickCount();
	auto load_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
	cout << "*模型编译时间ms:" << load_time << endl;

	return true;
}

bool Yolov4Dectector::uninit() {
	return true;
}

/// <summary>
/// sigmoid函数
/// </summary>
/// <param name="x"></param>
/// <returns></returns>
double Yolov4Dectector::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

/// <summary>
/// 获得不同规格的锚框
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
/// 处理输入图像
/// </summary>
/// <param name="inframe"></param>
/// <param name="out_detectod_result"></param>
/// <returns></returns>
void Yolov4Dectector::process_frame(unsigned char* image_batch, float* x1_ptr,
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

	// -------- Step 5. Create infer request --------
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// -------- Step 6. Combine multiple input images as batch --------
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	//const size_t image_size = shape_size(model->input().get_shape());
	//cout << "input_tensor:element_size:" << image_size << endl;
	_model_input_bs = input_tensor.get_shape()[0];
	_model_input_c = input_tensor.get_shape()[1];
	_model_input_h = input_tensor.get_shape()[2];
	_model_input_w = input_tensor.get_shape()[3];

	//计算原始图片和输入图片的比例
	_org_h_scale = float(_org_h) / _model_input_h;
	_org_w_scale = float(_org_w) / _model_input_w;

	cout << "input_tensor bs:" << _model_input_bs << endl;
	cout << "input_tensor c:" << _model_input_c << endl;
	cout << "input_tensor h:" << _model_input_h << endl;
	cout << "input_tensor w:" << _model_input_w << endl;

	float* blob_data = input_tensor.data<float>();
	//input，batch_size==1
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3, image_batch + i * _org_h * _org_w * 3);
		//cv::imwrite("org1.jpg",sourceImage);

		cv::resize(sourceImage, sourceImage, cv::Size(_model_input_w, _model_input_h));
		//cvtColor(blob_image, blob_image, COLOR_BGR2RGB);
		sourceImage.convertTo(sourceImage, CV_32FC3, 1 / 255.0f);
		//memcpy(blob_data + i * 3 * _model_input_h * _model_input_w, sourceImage.data, 3 * _model_input_h * _model_input_w * sizeof(float));

		//更新input_blob数据
		std::vector<cv::Mat>channles(3);
		cv::split(sourceImage, channles);
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w, channles[0].data, _model_input_h * _model_input_w * sizeof(float));
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w + _model_input_h * _model_input_w, channles[1].data, _model_input_h * _model_input_w * sizeof(float));
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w + 2 * _model_input_h * _model_input_w, channles[2].data, _model_input_h * _model_input_w * sizeof(float));
	}

	// -------- Step 9. Do asynchronous inference --------
	auto infer_begintime = cv::getTickCount();
	infer_request.set_input_tensor(input_tensor);
	//infer
	infer_request.infer();

	auto infer_endtime = cv::getTickCount();
	auto infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
	cout << "*推理时间ms:" << infer_time << endl;

	// -------- Step 10.  dispose output --------
	//output， bs==1
	//if bs>1, must be vector< vector<Rect>> origin_rect;vector<vector<float>> origin_rect_cof;vector<vector<int>> origin_class_pred
	vector<Rect> origin_rect;
	vector<float> origin_rect_cof;
	vector<int> origin_class_pred;
	auto parse_begintime = cv::getTickCount();

	for (size_t i = 0; i < _model_output_nums; i++)
	{
		ov::Tensor output = infer_request.get_output_tensor(i);
		cout << "output shape_"<<i<<":" << output.get_shape() << endl;
		parse_yolov4(output, _cof_threshold, _model_input_h, _model_input_w, _class_nums, origin_rect, origin_rect_cof, origin_class_pred);
	}
	//加密狗
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
	//

	auto parse_endtime = cv::getTickCount();
	infer_time = (to_string)((parse_endtime - parse_begintime) * 1000 / getTickFrequency());
	cout << "*解析输出时间:" << infer_time << endl;

	//NMS
	auto nms_begintime = cv::getTickCount();

	vector<int> final_id;
	dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_IOU, final_id);

	auto nms_endtime = cv::getTickCount();
	infer_time = (to_string)((nms_endtime - nms_begintime) * 1000 / getTickFrequency());
	cout << "*nms转换时间:" << infer_time << endl;

	auto out_begintime = cv::getTickCount();

	int nums_boxs = final_id.size();
	int output_length = min(nums_boxs, buffer_size);

	memcpy(num_boxes_ptr, &output_length, sizeof(int));
	num_boxes_ptr++;

	int count_pic = 0;
	int num = 0;

	for (int i = 0; i < output_length; i++)
	{
		float x1 = origin_rect[final_id[i]].x * _org_w_scale;
		float y1 = origin_rect[final_id[i]].y * _org_h_scale;
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
	infer_time = (to_string)((out_endtime - parse_begintime) * 1000 / getTickFrequency());
	cout << "*后处理总时间:" << infer_time << endl;

}


void Yolov4Dectector::parse_yolov4(ov::Tensor& output, float cof_threshold,
	int input_h, int input_w,int class_nums,
	vector<Rect>& o_rect, vector<float>& o_rect_cof,vector<int>& o_class_pred)
{   
	//0.加密信息
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


	//1.获取输出的dims
	auto outputDims = output.get_shape();
	
	//锚框个数指定为3
	int anchor_n = 3;
	//2.获得图片尺寸，
	int output_size = outputDims[1];
	int net_grid_w = outputDims[2];
	int net_grid_h = outputDims[3];
	_class_nums = output_size / anchor_n - 5;

	//输出长度为11，4（anchors）+1(class)+6（pred）
	int output_size_per = output_size / anchor_n;
    
	//3.获取锚框
	vector<int> anchors = get_anchors(net_grid_w);
	
	//4.获取blob数据
	const float* output_buffer = output.data<const float>();

	//5.计算置信度cofid和rect
	int anchors_position = 0;
	int count = 0;
	_stride_w = float(input_w) / net_grid_w;
	_stride_h = float(input_h) / net_grid_h;
	
	//head格式[bs,boxnums,nx,ny,[x,y,w,h,cof,cls1,cls2...cls80]]
	for (int n = 0; n < anchor_n; n++)
	{
		//判定通道数据的起始点
		anchors_position = n * output_size_per * net_grid_w * net_grid_h;
		count = 0;

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
				int x_index = anchors_position + 0 * net_grid_w * net_grid_h + count;
				int y_index = anchors_position + 1 * net_grid_w * net_grid_h + count;
				int w_index = anchors_position + 2 * net_grid_w * net_grid_h + count;
				int h_index = anchors_position + 3 * net_grid_w * net_grid_h + count;
				int conf_index = anchors_position + 4 * net_grid_w * net_grid_h + count;

				//计算conf，阈值过虑
				double box_prob = sigmoid(output_buffer[conf_index]);

				//找到分类中最大元素的位置
				float max_prob = 0;
				float class_conf = -1000;
				int idx = 0;
				//for (int t = 0; t < _class_nums; t++)
				//{
				//	auto temp = output_buffer[anchors_position + (t+5) * net_grid_w * net_grid_h + count];
				//	temp = sigmoid(temp);
				//	if (temp > max_prob)
				//	{
				//		max_prob = temp;
				//		idx = t;
				//	}
				//}
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

				//计算cof，等于max_prob * box_prob
				float cof = class_conf * box_prob;
				//对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
				if (cof < _cof_threshold)
				{
					count++;
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

				float center_x = output_buffer[x_index];
				float center_y = output_buffer[y_index];
				float center_w = output_buffer[w_index];
				float center_h = output_buffer[h_index];
				count++;

				//计算类别
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
				//计算rect
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
