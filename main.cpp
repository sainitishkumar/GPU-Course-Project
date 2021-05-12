// !nvcc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -I/usr/include/opencv2 main.cu
// !nvcc `pkg-config opencv --cflags --libs` -I/usr/include/opencv2 main.cu

#include <iostream>
#include <stdio.h>
#include <vector>
#include <stack>
#include <queue>
#include <math.h>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

typedef vector<float> vec_float;
typedef vector<int> vec_int;

void derivativesSrad(vec_float d_img, int rows, int cols, int totalSize, vec_int neighborN, vec_int neighborE, vec_int neighborW, vec_int neighborS, vec_float d_derN, vec_float d_derE, vec_float d_derW, vec_float d_derS, vec_float d_c, float q0Squared, float gamma) {
	
	for(int id=0; id<totalSize; id++) {
		int r = id/cols;
		int c = id%cols;
		float currval = d_img[id];
		
		float derivative_N = d_img[neighborN[r]*cols + c] - currval;
		float derivative_S = d_img[neighborS[r]*cols + c] - currval;
		float derivative_W = d_img[r*cols + neighborW[c]] - currval;
		float derivative_E = d_img[r*cols + neighborE[c]] - currval;

		float d_G2 = (derivative_N*derivative_N + derivative_S*derivative_S + derivative_W*derivative_W + derivative_E*derivative_E) / (currval*currval);
		float d_L = (derivative_N + derivative_S + derivative_W + derivative_E) / currval;
		
		float num = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L));
		float temp = 1 + (0.25*d_L);
		float qSquared = num/(temp*temp);
		temp = (qSquared-q0Squared) / (q0Squared * (1+q0Squared));
		
		// lee filter
		float k_lee = 1.0 / (1.0+temp);

		if (k_lee < 0) {
			k_lee = 0;
		} else if (k_lee > 1) {
			k_lee = 1;
		}

		d_derN[id] = derivative_N; 
		d_derS[id] = derivative_S; 
		d_derW[id] = derivative_W; 
		d_derE[id] = derivative_E;
		d_c[id] = k_lee;
	}
}

void srad(vec_float d_img, int rows, int cols, int totalSize, vec_int neighborN, vec_int neighborE, vec_int neighborW, vec_int neighborS, vec_float d_derN, vec_float d_derE, vec_float d_derW, vec_float d_derS, vec_float d_c, float gamma) {

	for(int id=0; id<totalSize; id++) {
		int r = id/cols;
		int c = id%cols;
		
		float d_cN = d_c[neighborN[r]*cols + c];
		float d_cS = d_c[neighborS[r]*cols + c];
		float d_cW = d_c[r*cols + neighborW[c]];
		float d_cE = d_c[r*cols + neighborE[c]];
		
		float d_D = d_cN*d_derN[id] + d_cS*d_derS[id] + d_cW*d_derW[id] + d_cE*d_derE[id];
		d_img[id] += 0.25*gamma*d_D;
	}
}

void derivativesOsrad(vec_float d_img, int rows, int cols, int totalSize, vec_int neighborN, vec_int neighborE, vec_int neighborW, vec_int neighborS, vec_float d_derN, vec_float d_derE, vec_float d_derW, vec_float d_derS, vec_float d_c, float q0Squared, float gamma) {

	for(int id=0; id<totalSize; id++) {
		int r = id/cols;
		int c = id%cols;
		float currval = d_img[id];
		
		float derivative_N = d_img[neighborN[r]*cols + c] - currval;
		float derivative_S = d_img[neighborS[r]*cols + c] - currval;
		float derivative_W = d_img[r*cols + neighborW[c]] - currval;
		float derivative_E = d_img[r*cols + neighborE[c]] - currval;

		float d_G2 = (derivative_N*derivative_N + derivative_S*derivative_S + derivative_W*derivative_W + derivative_E*derivative_E) / (currval*currval);
		float d_L = (derivative_N + derivative_S + derivative_W + derivative_E) / currval;
		
		float num = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L));
		float temp = 1 + (0.25*d_L);
		float qSquared = num/(temp*temp);
		temp = (qSquared-q0Squared) / (q0Squared * (1+q0Squared));
		
		// lee filter
		// float k_lee = 1.0 / (1.0+temp);
		float k_lee = exp(-1*temp);

		if (k_lee < 0) {
			k_lee = 0;
		} else if (k_lee > 1) {
			k_lee = 1;
		}

		d_derN[id] = derivative_N;
		d_derS[id] = derivative_S;
		d_derW[id] = derivative_W;
		d_derE[id] = derivative_E;
		d_c[id] = k_lee;
	}
}

void osrad(vec_float d_img, int rows, int cols, int totalSize, vec_int neighborN, vec_int neighborE, vec_int neighborW, vec_int neighborS, vec_float d_derN, vec_float d_derE, vec_float d_derW, vec_float d_derS, vec_float d_c, float gamma, int ctang) {

	for(int id=0; id<totalSize; id++) {
		int r = id/cols;
		int c = id%cols;
		
		float d_cN = d_c[neighborN[r]*cols + c];
		float d_cS = d_c[neighborS[r]*cols + c];
		float d_cW = d_c[r*cols + neighborW[c]];
		float d_cE = d_c[r*cols + neighborE[c]];
		
		d_cS = d_cS+ctang;
		d_cE = d_cE+ctang;
		d_cN = d_c[id]+ctang;
		d_cW = d_c[id]+ctang;

		float d_D = d_cN*d_derN[id] + d_cS*d_derS[id] + d_cW*d_derW[id] + d_cE*d_derE[id];

		d_img[id] += 0.25*gamma*d_D;
	}
}

void copySums(vec_float d_img, vec_float sum1, vec_float sum2, int totalSize) {
	for(int id=0; id<totalSize; id++) {
			sum1[id] = d_img[id];
			sum2[id] = d_img[id]*d_img[id];
	}
}

float epower_functor(int x) {
	return expf(x/255);
}

float log_functor(int x) {
	return round(log(x)*255*1000.0)/1000.0;
}

float square_functor(int x) {
	return x*x;
}

int main(int argc, char **argv) {
	// read img from path
	auto image_path = argv[1];
    Mat mat = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    if(mat.empty()) {
        cout << "Could not read the image: " << image_path << endl;
        return 1;
    }
    std::vector<float> h_img;
	if (mat.isContinuous()) {
		h_img.assign(mat.data, mat.data + mat.total()*mat.channels());
	} else {
		for (int i = 0; i < mat.rows; ++i) {
			h_img.insert(h_img.end(), mat.ptr<float>(i), mat.ptr<float>(i)+mat.cols*mat.channels());
		}
	}
	
	// params
	int rows = mat.rows;
	int cols = mat.cols;
	int totalSize = rows*cols;

	vector<int> neighborN(rows, 0);
	vector<int> neighborS(rows, 0);
	vector<int> neighborE(cols, 0);
	vector<int> neighborW(cols, 0);
	
	for (int i=0; i<rows; i++) {
		neighborN[i] = i-1;
		neighborS[i] = i+1;
	}
	for (int j=0; j<cols; j++) {
		neighborW[j] = j-1;
		neighborE[j] = j+1;
	}
	
	neighborN[0]=0;
	neighborS.back()=rows-1;
	neighborW[0]=0;
	neighborE.back()=cols-1;

	// setting device variables for calculation
	vec_float d_img = h_img;
	std::transform(d_img.begin(), d_img.end(), d_img.begin(), epower_functor);
	
	// derivatives
	vec_float d_derN(totalSize, -1);
	vec_float d_derE(totalSize, -1);
	vec_float d_derW(totalSize, -1);
	vec_float d_derS(totalSize, -1);
	
	// k_lee coefficient
	vec_float d_c(totalSize, 0);

	int num_iter = stoi(argv[2]);
	float gamma = stof(argv[3]);
	int method = stoi(argv[4]);

	// following jacobi scheme
	while(num_iter--) {

		vec_float sumArr(totalSize, 0);
		vec_float sumSquared(totalSize, 0);
		copySums(d_img, sumArr, sumSquared, totalSize);
		
		float sumVal = std::accumulate(sumArr.begin(), sumArr.end(), 0);
		float sumSq = std::accumulate(sumSquared.begin(), sumSquared.end(), 0);

		float mean, meanSquared, variance, stdev, q0Squared;

		mean = (float)sumVal/(float)totalSize;
		meanSquared = (float)sumSq/(float)totalSize;
		variance = meanSquared - (mean*mean);
		stdev = sqrt(variance);
		q0Squared = (float)stdev/((float)(mean*mean));
		
		if(method == 1) {
			// finding deviation coefficients first for all pixels
			derivativesSrad(d_img, rows, cols, totalSize, 
						neighborN, neighborE, neighborW, neighborS, 
						d_derN, d_derE, d_derW, d_derS, d_c, q0Squared, gamma);
			// global barrier as we have two kernel calls
			// then update image
			srad(d_img, rows, cols, totalSize, 
				neighborN, neighborE, neighborW, neighborS, 
				d_derN, d_derE, d_derW, d_derS, d_c, gamma);
		} else if(method == 2) {
			int ctang = stoi(argv[5]);
			derivativesOsrad(d_img, rows, cols, totalSize, 
						neighborN, neighborE, neighborW, neighborS, 
						d_derN, d_derE, d_derW, d_derS, d_c, q0Squared, gamma);

			osrad(d_img, rows, cols, totalSize, 
				neighborN, neighborE, neighborW, neighborS, 
				d_derN, d_derE, d_derW, d_derS, d_c, gamma, ctang);
		}
	}
	
	std::transform(d_img.begin(), d_img.end(), d_img.begin(), log_functor);
	vec_float img = d_img;

	cv::Mat imgMat(rows, cols, CV_32FC1);
	memcpy(imgMat.data, img.data(), img.size()*sizeof(float));

	if(method==1)
		imwrite(argv[5], imgMat);
	if(method==2)
		imwrite(argv[6], imgMat);
	return 0;
}















