#pragma once
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


void matU8ToBlob_t(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);