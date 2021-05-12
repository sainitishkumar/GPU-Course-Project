// !nvcc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -I/usr/include/opencv2 main.cu
// !nvcc `pkg-config opencv --cflags --libs` -I/usr/include/opencv2 main.cu

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <vector>
#include <stack>
#include <queue>
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#define BLOCKSIZE 1024

typedef thrust::device_vector<float> dvec_float;
typedef thrust::device_vector<int> dvec_int;

typedef thrust::host_vector<float> hvec_float;
typedef thrust::host_vector<int> hvec_int;

using namespace std;
using namespace cv;

struct epower_functor {
    __host__ __device__
    float operator()(const float& x) const { 
        return expf(x/255);
    }
};

struct square_functor {
    __host__ __device__
    float operator()(const float& x) const { 
        return x*x;
    }
};

struct log_functor {
    __host__ __device__
    float operator()(const float& x) const { 
        return round(log(x)*255*1000.0)/1000.0;
    }
};

__global__ void derivativesSrad(float *d_img, int rows, int cols, int totalSize, int *d_neighN, int *d_neighE, int *d_neighW, int *d_neighS, float *d_derN, float *d_derE, float *d_derW, float *d_derS, float *d_c, float q0Squared, float gamma) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < totalSize) {
		int r = id/cols;
		int c = id%cols;
		float currval = d_img[id];
		
		float derivative_N = d_img[d_neighN[r]*cols + c] - currval;
		float derivative_S = d_img[d_neighS[r]*cols + c] - currval;
		float derivative_W = d_img[r*cols + d_neighW[c]] - currval;
		float derivative_E = d_img[r*cols + d_neighE[c]] - currval;

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

__global__ void srad(float *d_img, int rows, int cols, int totalSize, int *d_neighN, int *d_neighE, int *d_neighW, int *d_neighS, float *d_derN, float *d_derE, float *d_derW, float *d_derS, float *d_c, float gamma) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < totalSize) {
		int r = id/cols;
		int c = id%cols;
		
		float d_cN = d_c[d_neighN[r]*cols + c];
		float d_cS = d_c[d_neighS[r]*cols + c];
		float d_cW = d_c[r*cols + d_neighW[c]];
		float d_cE = d_c[r*cols + d_neighE[c]];
		
		float d_D = d_cN*d_derN[id] + d_cS*d_derS[id] + d_cW*d_derW[id] + d_cE*d_derE[id];
		d_img[id] += 0.25*gamma*d_D;
	}
}

__global__ void derivativesOsrad(float *d_img, int rows, int cols, int totalSize, int *d_neighN, int *d_neighE, int *d_neighW, int *d_neighS, float *d_derN, float *d_derE, float *d_derW, float *d_derS, float *d_c, float q0Squared, float gamma) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < totalSize) {
		int r = id/cols;
		int c = id%cols;
		float currval = d_img[id];
		
		float derivative_N = d_img[d_neighN[r]*cols + c] - currval;
		float derivative_S = d_img[d_neighS[r]*cols + c] - currval;
		float derivative_W = d_img[r*cols + d_neighW[c]] - currval;
		float derivative_E = d_img[r*cols + d_neighE[c]] - currval;

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

__global__ void osrad(float *d_img, int rows, int cols, int totalSize, int *d_neighN, int *d_neighE, int *d_neighW, int *d_neighS, float *d_derN, float *d_derE, float *d_derW, float *d_derS, float *d_c, float gamma, int ctang) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < totalSize) {
		int r = id/cols;
		int c = id%cols;
		
		float d_cN = d_c[d_neighN[r]*cols + c];
		float d_cS = d_c[d_neighS[r]*cols + c];
		float d_cW = d_c[r*cols + d_neighW[c]];
		float d_cE = d_c[r*cols + d_neighE[c]];
		
		d_cS = d_cS+ctang;
		d_cE = d_cE+ctang;
		d_cN = d_c[id]+ctang;
		d_cW = d_c[id]+ctang;

		float d_D = d_cN*d_derN[id] + d_cS*d_derS[id] + d_cW*d_derW[id] + d_cE*d_derE[id];

		d_img[id] += 0.25*gamma*d_D;
	}
}

// __global__ void printkernel(float *d_img, int rows, int cols, int totalSize) {
// 	for(int i=0; i<rows; i++) {
// 		for(int j=0; j<cols; j++) {
// 			printf("%f ", d_img[i*cols+j]);
// 		}
// 		printf("\n");
// 	}
// }

__global__ void copykernel(float *d_img, float *sum1, float *sum2, int totalSize) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < totalSize) {
		sum1[id] = d_img[id];
		sum2[id] = d_img[id]*d_img[id];
	}
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

	vector<float> neighborN(rows, 0);
	vector<float> neighborS(rows, 0);
	vector<float> neighborE(cols, 0);
	vector<float> neighborW(cols, 0);
	
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
	dvec_float d_img(totalSize);
	thrust::copy(h_img.begin(), h_img.end(), d_img.begin());
	thrust::transform(d_img.begin(), d_img.end(), d_img.begin(), epower_functor());

	// neighbors
	dvec_int d_neighN{neighborN};
	dvec_int d_neighS{neighborS};
	dvec_int d_neighW{neighborW};
	dvec_int d_neighE{neighborE};
	
	// derivatives
	dvec_float d_derN(totalSize, -1);
	dvec_float d_derE(totalSize, -1);
	dvec_float d_derW(totalSize, -1);
	dvec_float d_derS(totalSize, -1);
	
	// k_lee coefficient
	dvec_float d_c(totalSize, 0);

	int num_iter = stoi(argv[2]);
	float gamma = stof(argv[3]);
	int method = stoi(argv[4]);

	int nblocks = ceil((float)totalSize / BLOCKSIZE);

	// following jacobi scheme
	while(num_iter--) {

		dvec_float sumArr(totalSize, 0);
		dvec_float sumSquared(totalSize, 0);

		copykernel<<<nblocks, BLOCKSIZE>>>(thrust::raw_pointer_cast(&d_img[0]), thrust::raw_pointer_cast(&sumArr[0]), thrust::raw_pointer_cast(&sumSquared[0]), totalSize);
		cudaDeviceSynchronize();
		
		float sumVal = thrust::reduce(sumArr.begin(), sumArr.end());
		float sumSq = thrust::reduce(sumSquared.begin(), sumSquared.end());

		float mean, meanSquared, variance, stdev, q0Squared;

		mean = (float)sumVal/(float)totalSize;
		meanSquared = (float)sumSq/(float)totalSize;
		variance = meanSquared - (mean*mean);
		stdev = sqrt(variance);
		q0Squared = (float)stdev/((float)(mean*mean));
		
		if(method == 1) {
			// finding deviation coefficients first for all pixels
			derivativesSrad<<<nblocks, BLOCKSIZE>>>(thrust::raw_pointer_cast(&d_img[0]), rows, cols, totalSize, 
												thrust::raw_pointer_cast(&d_neighN[0]), thrust::raw_pointer_cast(&d_neighE[0]), thrust::raw_pointer_cast(&d_neighW[0]), thrust::raw_pointer_cast(&d_neighS[0]), 
												thrust::raw_pointer_cast(&d_derN[0]), thrust::raw_pointer_cast(&d_derE[0]), thrust::raw_pointer_cast(&d_derW[0]), thrust::raw_pointer_cast(&d_derS[0]), thrust::raw_pointer_cast(&d_c[0]), q0Squared, gamma);
			// global barrier as we have two kernel calls
			// then update image
			srad<<<nblocks, BLOCKSIZE>>>(thrust::raw_pointer_cast(&d_img[0]), rows, cols, totalSize, 
												thrust::raw_pointer_cast(&d_neighN[0]), thrust::raw_pointer_cast(&d_neighE[0]), thrust::raw_pointer_cast(&d_neighW[0]), thrust::raw_pointer_cast(&d_neighS[0]), 
												thrust::raw_pointer_cast(&d_derN[0]), thrust::raw_pointer_cast(&d_derE[0]), thrust::raw_pointer_cast(&d_derW[0]), thrust::raw_pointer_cast(&d_derS[0]), thrust::raw_pointer_cast(&d_c[0]), gamma);
		} else if(method == 2) {
			int ctang = stoi(argv[5]);

			derivativesOsrad<<<nblocks, BLOCKSIZE>>>(thrust::raw_pointer_cast(&d_img[0]), rows, cols, totalSize, 
												thrust::raw_pointer_cast(&d_neighN[0]), thrust::raw_pointer_cast(&d_neighE[0]), thrust::raw_pointer_cast(&d_neighW[0]), thrust::raw_pointer_cast(&d_neighS[0]), 
												thrust::raw_pointer_cast(&d_derN[0]), thrust::raw_pointer_cast(&d_derE[0]), thrust::raw_pointer_cast(&d_derW[0]), thrust::raw_pointer_cast(&d_derS[0]), thrust::raw_pointer_cast(&d_c[0]), q0Squared, gamma);

			osrad<<<nblocks, BLOCKSIZE>>>(thrust::raw_pointer_cast(&d_img[0]), rows, cols, totalSize, 
												thrust::raw_pointer_cast(&d_neighN[0]), thrust::raw_pointer_cast(&d_neighE[0]), thrust::raw_pointer_cast(&d_neighW[0]), thrust::raw_pointer_cast(&d_neighS[0]), 
												thrust::raw_pointer_cast(&d_derN[0]), thrust::raw_pointer_cast(&d_derE[0]), thrust::raw_pointer_cast(&d_derW[0]), thrust::raw_pointer_cast(&d_derS[0]), thrust::raw_pointer_cast(&d_c[0]), gamma, ctang);
		}
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();

	thrust::transform(d_img.begin(), d_img.end(), d_img.begin(), log_functor());

	thrust::host_vector<float> img(totalSize, 0);
	thrust::copy(d_img.begin(), d_img.end(), img.begin());

	cv::Mat imgMat(rows, cols, CV_32FC1);
	memcpy(imgMat.data, img.data(), img.size()*sizeof(float));
	if(method==1)
		imwrite(argv[5], imgMat);
	if(method==2)
		imwrite(argv[6], imgMat);
	return 0;
}















