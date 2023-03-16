#include"detector.h";
#include"LicenseChecker.h"

UnetDectector::UnetDectector() {};
UnetDectector::~UnetDectector() {};

//std::string fileNameWithoutExt(const std::string& fpath) {
//	return fpath.substr(0, std::min<size_t>(fpath.size(), fpath.rfind('.')));
//}

/// <summary>
/// ��ʼ������
/// </summary>
/// <param name="device">�豸="CPU""GPU"</param>
/// <param name="xml_path">xml�ļ�·��</param>
bool UnetDectector::InitializeDetector(string device, string onnx_path)
{
	_onnx_path = onnx_path;
	_device = device;

	// -------- Step 1. Initialize OpenVINO Runtime Core --------
	//ov::Core core;
	//auto  devices = core.get_available_devices();
	//for (auto iter = devices.begin(); iter != devices.end(); iter++)
	//{
	//	cout << "��ǰ�����豸���ƣ�" << iter->c_str() << endl;
	//}

	// -------- Step 2. Read a model --------
	try
	{
		auto load_begintime = cv::getTickCount();

		model = core.read_model(_onnx_path);
		//cout << "input_names_size:" << model->input().get_names().size() << endl;
		//cout << "input_tensor:" << model->input().get_tensor() << endl;
		//cout << "input_name:" << model->input().get_any_name() << endl;

		cout << "input_type:" << model->input().get_element_type() << endl;
		cout << "input_shape:" << model->input().get_shape() << endl;

		cout << "output_type0:" << model->get_output_element_type(0) << endl;
		cout << "output_shape0:" << model->get_output_shape(0) << endl;
		//cout << "output_shape:" << model->output().get_shape() << endl;

		//cout << "output_shape1:" << model->get_output_shape(1) << endl;
		//cout << "output_shape2:" << model->get_output_shape(2) << endl;

		//cout << "output_type0:" << model->get_output_element_type(0) << endl;
		//cout << "output_type1:" << model->get_output_element_type(1) << endl;
		//cout << "output_type2:" << model->get_output_element_type(2) << endl;

		//cout << "output_names_size:" << model->output().get_names().size() << endl;
		//cout << "output_type:" << model->output().get_element_type() << endl;
		//cout << "output_shape:" << model->output().get_shape() << endl;
		//cout << "output_tensor:" << model->output().get_tensor() << endl;
		//cout << "output_name:" << model->output().get_any_name() << endl;

		auto load_endtime = cv::getTickCount();
		auto infer_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
		cout << "*ģ�ͼ��سɹ���" << device << "  *����ʱ��ms:" << infer_time << endl;


	}
	catch (const std::exception& ex)
	{
		cout << "*ģ�ͼ���ʧ�ܣ�" << device << endl;
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

/// <summary>
/// ������̣��õ����
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
	//��ȡԭʼ�������
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;

	//����ģ�͵�batch size
	ov::set_batch(model, _batch_size);

	// -------- Step 4. Loading model to the device --------
	auto load_begintime = cv::getTickCount();
	ov::CompiledModel compiled_model = core.compile_model(_onnx_path, _device);
	auto load_endtime = cv::getTickCount();
	auto load_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
	cout << "*ģ�ͱ���ʱ��ms:" << load_time << endl;

	// -------- Step 5. Create infer request --------
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// -------- Step 6. Combine multiple input images as batch --------
	ov::Tensor input_tensor = infer_request.get_input_tensor();


	const size_t image_size = shape_size(model->input().get_shape());
	cout << "input_tensor:element_size:" << image_size << endl;
	
	_model_input_h = input_tensor.get_shape()[2];
	_model_input_w = input_tensor.get_shape()[3];

	cout << "input_tensor h:" << _model_input_h << endl;
	cout << "input_tensor w:" << _model_input_w << endl;
	
	float* blob_data = input_tensor.data<float>();
	
	vector<Mat> batch_Pic;
	//ѭ������ͼƬ����������Ԥ��
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3, image_batch + i * _org_h * _org_w * 3);
		cv::imwrite("org1.jpg",sourceImage);

		cv::resize(sourceImage, sourceImage, cv::Size(_model_input_w, _model_input_h));
		//cvtColor(blob_image, blob_image, COLOR_BGR2RGB);
		sourceImage.convertTo(sourceImage, CV_32FC3, 1 / 255.0f);
		//memcpy(blob_data + i * 3 * _model_input_h * _model_input_w, sourceImage.data, 3 * _model_input_h * _model_input_w * sizeof(float));

		//����input_blob����
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
		cout << "*����ʱ��ms:" << infer_time << endl;

		// -------- Step 10.  dispose output --------
		ov::Tensor output = infer_request.get_output_tensor();
		cout << "output shape:" << output.get_shape() << endl;
		batch_size = output.get_shape()[0];
		_model_output_class = output.get_shape()[1];
		_model_output_h = output.get_shape()[2];
		_model_output_w = output.get_shape()[3];

		parse_Unet(output, batch_Pic);
	}

	//���ܹ�
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

	//�������  _org_h*_org_w * _output_class* _batch_size
	int picSize_per = _org_h * _org_w;
	int bs_class_per = picSize_per * _model_output_class;

	//unsigned char* result = new unsigned char[batch_size_class_per * _batch_size];
	//��batch��ÿ��ͼƬ���뵽ͼƬָ��0,1,2
	for (size_t bs_i = 0; bs_i < _batch_size; bs_i++)
	{
		for (size_t pic_i = 0; pic_i < _model_output_class; pic_i++)
		{
			//������batch��ͼƬ���ص�ͼƬָ����
			memcpy(prediction_batch + bs_i * bs_class_per + picSize_per * pic_i, batch_Pic[bs_i * _model_output_class + pic_i].data, picSize_per * sizeof(unsigned char));
		}
	}
}


/// <summary>
/// ����output��������
/// </summary>
/// <param name="blob">����blob</param>
/// <param name="cnts">���uchar*</param>
/// <param name="width">ԭͼ���</param>
/// <param name="height">ԭͼ�߶�</param>
void UnetDectector::parse_Unet(ov::Tensor& output, vector<Mat>& allPic)
{
	const float* batchdata = output.data<const float>();

	UnetDectector::getResult_Unet(batchdata, allPic, _model_output_class, _model_output_w, _model_output_h);
}

/// <summary>
/// ���blob�������uchar*
/// </summary>
/// <param name="output_buffer">const float* ����</param>
/// <param name="cnts">uchar*���</param>
/// <param name="output_class">������</param>
/// <param name="output_w">ģ��������</param>
/// <param name="output_h">ģ������߶�</param>
void UnetDectector::getResult_Unet(const float* output_buffer, vector<Mat>& allPic, int output_class, int output_w, int output_h)
{
	int img_size = output_w * output_h;
	int all_img_size = output_class * img_size;
	uchar* cnts =new unsigned char[all_img_size];
	uchar* newcnts = cnts;
	//scale
	for (size_t i = 0; i < all_img_size; i++)
	{
		uchar pixel = *(output_buffer + i) * 255;
		*newcnts = pixel;
		newcnts++;
	}

	//�������
	int output_size = _org_w * _org_h;
	int output_class_last = output_class;
	for (size_t i_class = 0; i_class < output_class_last; i_class++)
	{
		//Mat resultImage = Mat(output_h, output_w, CV_8UC1, Scalar(0));
		vector <uchar> mask;
		mask.resize(img_size);
		cout << "ͼƬ��" << i_class << endl;

		for (size_t h = 0; h < img_size; h++)
		{
			mask[h] = 0;
			float pixel = *(cnts + img_size * i_class + h);
			mask[h] = pixel;
		}
		unsigned char* maskImage = mask.data();
		cv::Mat mask_mat = cv::Mat(output_h, output_w, CV_8UC1);
		mask_mat.data = maskImage;

		cv::Mat mask_mat_resized;
		cv::resize(mask_mat, mask_mat_resized, cv::Size(_org_w, _org_h));

		allPic.push_back(mask_mat_resized);
	}
	int test = true;
	if (test)
	{
		//��ӡ��֤���
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




