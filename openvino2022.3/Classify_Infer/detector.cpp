#include"detector.h";
#include"LicenseChecker.h"

ClassifyClass::ClassifyClass() {};

ClassifyClass::~ClassifyClass() {};

//std::string fileNameWithoutExt(const std::string& fpath) {
//	return fpath.substr(0, std::min<size_t>(fpath.size(), fpath.rfind('.')));
//}

/// <summary>
/// 初始化网络
/// </summary>
/// <param name="device">设备="CPU""GPU"</param>
/// <param name="xml_path">xml文件路径</param>
bool ClassifyClass::InitializeDetector(string device, string xml_path)
{
	_onnxpath = xml_path;
	_device = device;

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
	//ov::Core core;
	//auto  devices = core.get_available_devices();
	//for (auto iter = devices.begin(); iter != devices.end(); iter++)
	//{
	//	cout << "当前可用设备名称：" << iter->c_str() << endl;
	//}

    // -------- Step 2. Read a model --------
	try
	{
		auto load_begintime = cv::getTickCount();
		 
		model = core.read_model(_onnxpath);
		 cout << "input_names_size:" << model->input().get_names().size() << endl;
		 cout << "input_type:" << model->input().get_element_type() << endl;
		 cout << "input_shape:" << model->input().get_shape() << endl;
		 cout << "input_tensor:" << model->input().get_tensor() << endl;
		 cout << "input_name:" << model->input().get_any_name() << endl;

		 cout << "output_names_size:" << model->output().get_names().size() << endl;
		 cout << "output_type:" << model->output().get_element_type() << endl;
		 cout << "output_shape:" << model->output().get_shape() << endl;
		 cout << "output_tensor:" << model->output().get_tensor() << endl;
		 cout << "output_name:" << model->output().get_any_name() << endl;

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
	const ov::Layout tensor_layout{"NCHW"};
	input_info.tensor().set_element_type(ov::element::f32).set_layout(tensor_layout);
	input_info.preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
	//input_info.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
	
	// 3) Here we suppose model has 'NCHW' layout for input
	input_info.model().set_layout("NCHW");
	
	// 4) output() with no args assumes a model has a single result
	// - precision of tensor is supposed to be 'f32'
	ppp.output().tensor().set_element_type(ov::element::f32);

	// 5) Once the build() method is called, the pre(post)processing steps
	// for layout and precision conversions are inserted automatically
	model = ppp.build();

	return true;
}

bool ClassifyClass::uninit() {
	return true;
}


void ClassifyClass::process_frame(unsigned char* image_batch, float* prediction_batch,
	int width, int height, int training_size, int batch_size)
{
	//获取原始输入参数
	_org_w = width;
	_org_h = height;
	_batch_size = batch_size;

	ov::set_batch(model, _batch_size);

	// -------- Step 4. Loading model to the device --------
	auto load_begintime = cv::getTickCount();
	ov::CompiledModel compiled_model = core.compile_model(_onnxpath, _device);
	auto load_endtime = cv::getTickCount();
	auto load_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
	cout << "*模型编译时间ms:" << load_time << endl;

	// -------- Step 5. Create infer request --------
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// -------- Step 6. Combine multiple input images as batch --------
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	cout << endl;
	cout << "input_tensor:shape:" << input_tensor.get_shape() << endl;
	cout << "input_tensor:element_type:" << input_tensor.get_element_type() << endl;

	const size_t image_size = shape_size(model->input().get_shape());
	cout << "input_tensor:element_size:" << image_size << endl;
	_model_input_h = input_tensor.get_shape()[2];
	_model_input_w = input_tensor.get_shape()[3];
	cout << "input_tensor h:" << _model_input_h << endl;
	cout << "input_tensor w:" << _model_input_w << endl;

	float* blob_data = input_tensor.data<float>();
	//循环所有图片，进行推理预测
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3, image_batch + i * _org_h * _org_w * 3);
		cv::imwrite("org1.jpg", sourceImage);
		Mat blob_image;
		cv::resize(sourceImage, sourceImage, cv::Size(_model_input_w, _model_input_h));
		//cvtColor(blob_image, blob_image, COLOR_BGR2RGB);
		sourceImage.convertTo(sourceImage, CV_32FC3, 1 / 255.0f);
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w, sourceImage.data, 3*_model_input_h * _model_input_w * sizeof(float));

		//更新input_blob数据
		/*std::vector<cv::Mat>channles(3);
		cv::split(sourceImage, channles);
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w, channles[0].data, _model_input_h * _model_input_w * sizeof(float));
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w + _model_input_h * _model_input_w, channles[1].data, _model_input_h * _model_input_w * sizeof(float));
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w + 2 * _model_input_h * _model_input_w, channles[2].data, _model_input_h * _model_input_w * sizeof(float));*/
	}
	//memcpy(blob_data, image_batch, image_size * sizeof(float));

	// -------- Step 9. Do asynchronous inference --------
	auto infer_begintime = cv::getTickCount();
	infer_request.set_input_tensor(input_tensor);
	infer_request.infer();

	auto infer_endtime = cv::getTickCount();
	auto infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
	cout << "*推理时间ms:" << infer_time << endl;

	// -------- Step 10.  dispose output --------
	vector<float> batch_res;
	ov::Tensor output = infer_request.get_output_tensor();
	parse_classify(output, batch_res);

	//cv::Mat img_temp = cv::Mat(_org_h,_org_w, CV_8UC3);
	//img_temp.data = image_batch;
	//cv::imwrite("t1.jpg",img_temp);

	// -------- Step 8. Do asynchronous inference --------
	//auto infer_begintime = cv::getTickCount();
	//size_t num_iterations = 10;
	//size_t cur_iteration = 0;
	//std::condition_variable condVar;
	//std::mutex mutex;
	//std::exception_ptr exception_var;

	//infer_request.set_callback([&](std::exception_ptr ex) {
	//	std::lock_guard<std::mutex> l(mutex);
	//	if (ex) {
	//		exception_var = ex;
	//		condVar.notify_all();
	//		return;
	//	}

	//	cur_iteration++;
	//	//slog::info << "Completed " << cur_iteration << " async request execution" << slog::endl;
	//	if (cur_iteration < num_iterations) {
	//		// here a user can read output containing inference results and put new
	//		// input to repeat async request again
	//		infer_request.start_async();
	//	}
	//	else {
	//		// continue sample execution after last Asynchronous inference request
	//		// execution
	//		condVar.notify_one();
	//	}
	//	});

	// Start async request for the first time
	//slog::info << "Start inference (asynchronous executions)" << slog::endl;
	//infer_request.start_async();

	//// Wait all iterations of the async request
	//std::unique_lock<std::mutex> lock(mutex);
	//condVar.wait(lock, [&] {
	//	if (exception_var) {
	//		std::rethrow_exception(exception_var);
	//	}

	//	return cur_iteration == num_iterations;
	//	});

	////slog::info << "Completed async requests execution" << slog::endl;

	//auto infer_endtime = cv::getTickCount();
	//auto infer_time = (to_string)((infer_endtime - infer_endtime) * 1000 / getTickFrequency());
	//cout << "*推理时间ms:" << infer_time << endl;
	
	// -------- Step 9. Process output --------

	memcpy(prediction_batch, batch_res.data(), sizeof(float)* batch_res.size()*batch_size);
}

void ClassifyClass::process_frame_batch(unsigned char* image_batch, float* prediction_batch,
	int width, int height, int training_size, int batch_size)
{
	//获取原始输入参数
	_org_w = width;
	_org_h = height;
	_batch_size = batch_size;

	ov::set_batch(model, _batch_size);
	// -------- Step 4. Loading model to the device --------
	auto load_begintime = cv::getTickCount();
	ov::CompiledModel compiled_model = core.compile_model(_onnxpath, _device);
	auto load_endtime = cv::getTickCount();
	auto load_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
	cout << "*模型编译时间ms:" << load_time << endl;

	// -------- Step 5. Create infer request --------
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// -------- Step 6. Combine multiple input images as batch --------
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	cout << endl;
	cout << "input_tensor:shape:" << input_tensor.get_shape() << endl;
	cout << "input_tensor:element_type:" << input_tensor.get_element_type() << endl;

	const size_t image_size = shape_size(model->input().get_shape());
	cout << "input_tensor:element_size:" << image_size << endl;
	_model_input_h = input_tensor.get_shape()[2];
	_model_input_w = input_tensor.get_shape()[3];
	cout << "input_tensor h:" << _model_input_h << endl;
	cout << "input_tensor w:" << _model_input_w << endl;

	//std::memcpy(input_tensor.data<float>() ,image_batch,image_size);
	vector<float> batch_res;
	//循环所有图片，进行推理预测
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3, image_batch + i * _org_h * _org_w * 3);
		cv::imwrite("org1.jpg",sourceImage);
		Mat blob_image;
		cv::resize(sourceImage, sourceImage, cv::Size(_model_input_w, _model_input_h));
		//cvtColor(blob_image, blob_image, COLOR_BGR2RGB);
		sourceImage.convertTo(sourceImage, CV_32FC3, 1 / 255.0f);

		//更新input_blob数据
		float* blob_data = input_tensor.data<float>();
		std::vector<cv::Mat>channles(3);
		cv::split(sourceImage, channles);
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w, channles[0].data, _model_input_h * _model_input_w * sizeof(float));
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w + _model_input_h * _model_input_w, channles[1].data, _model_input_h * _model_input_w * sizeof(float));
		memcpy(blob_data + i * 3 * _model_input_h * _model_input_w + 2 * _model_input_h * _model_input_w, channles[2].data, _model_input_h * _model_input_w * sizeof(float));

		// -------- Step 9. Do asynchronous inference --------
		auto infer_begintime = cv::getTickCount();
		infer_request.set_input_tensor(input_tensor);
		infer_request.infer();

		auto infer_endtime = cv::getTickCount();
		auto infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		cout << "*推理时间ms:" << infer_time << endl;

		// -------- Step 7.  Prepare input --------
		ov::Tensor output = infer_request.get_output_tensor();

		mode_type = "Classify";
		if (mode_type=="Classify")
		{
			parse_classify(output, batch_res);

		}
		else if (mode_type == "Segmentor")
		{

		}
		else if (mode_type=="Yolo")
		{

		}
					
	}

	//cv::Mat img_temp = cv::Mat(_org_h,_org_w, CV_8UC3);
	//img_temp.data = image_batch;
	//cv::imwrite("t1.jpg",img_temp);

	// -------- Step 8. Do asynchronous inference --------
	//auto infer_begintime = cv::getTickCount();
	//size_t num_iterations = 10;
	//size_t cur_iteration = 0;
	//std::condition_variable condVar;
	//std::mutex mutex;
	//std::exception_ptr exception_var;

	//infer_request.set_callback([&](std::exception_ptr ex) {
	//	std::lock_guard<std::mutex> l(mutex);
	//	if (ex) {
	//		exception_var = ex;
	//		condVar.notify_all();
	//		return;
	//	}

	//	cur_iteration++;
	//	//slog::info << "Completed " << cur_iteration << " async request execution" << slog::endl;
	//	if (cur_iteration < num_iterations) {
	//		// here a user can read output containing inference results and put new
	//		// input to repeat async request again
	//		infer_request.start_async();
	//	}
	//	else {
	//		// continue sample execution after last Asynchronous inference request
	//		// execution
	//		condVar.notify_one();
	//	}
	//	});

	// Start async request for the first time
	//slog::info << "Start inference (asynchronous executions)" << slog::endl;
	//infer_request.start_async();

	//// Wait all iterations of the async request
	//std::unique_lock<std::mutex> lock(mutex);
	//condVar.wait(lock, [&] {
	//	if (exception_var) {
	//		std::rethrow_exception(exception_var);
	//	}

	//	return cur_iteration == num_iterations;
	//	});

	////slog::info << "Completed async requests execution" << slog::endl;

	//auto infer_endtime = cv::getTickCount();
	//auto infer_time = (to_string)((infer_endtime - infer_endtime) * 1000 / getTickFrequency());
	//cout << "*推理时间ms:" << infer_time << endl;

	// -------- Step 9. Process output --------

	memcpy(prediction_batch, batch_res.data(), sizeof(float) * batch_res.size() * batch_size);
}

void ClassifyClass::parse_classify(ov::Tensor& output, vector<float>& per_pic_res)
{
	int output_class = output.get_shape()[1];
	cout << "output shape:" << output.get_shape() << endl;
	//_model_output_class = output_class;
	int batch_size = output.get_shape()[0];

	const float* batchdata = output.data<const float>();

	for (size_t cls_i = 0; cls_i < output_class; cls_i++)
	{
		per_pic_res.push_back(*(batchdata + cls_i));
	}
}



