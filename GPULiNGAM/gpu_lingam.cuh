#ifndef GPU_LINGAM_CUH
#define GPU_LINGAM_CUH

#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 512

using namespace std;

class gpu_lingam{
private:
    int dims = 0;
    int samples = 0;
    bool verbose = true;
    vector<int> causal_order;
public:
    gpu_lingam();
    ~gpu_lingam();

    vector<int> fit(double* X, int n, int m);
    void set_verbose_mode(bool verbose);
    vector<int> get_causal_order();
};

__global__ void normalize_data(double* d_X, double* d_X_norm, int samples, int dims);
__global__ void compute_M(double* d_X_norm, double* d_M_list, int* d_U, int dims, int samples, int iter);
__global__ void block_compute_M(double* d_X_norm, double* d_M_list, int* d_U, int dims, int samples, int iter, int num_threads);
__global__ void update_order(int* d_U, int* d_causal_order, double* d_M_list, int dims, int iter);
__global__ void residualize_data(double* d_X, int* d_causal_order, int dims, int samples, int iter);
__device__ int max_index(double* array, int size);
__device__ void remove_element_by_value(int* array, int value, int size);
__device__ void normalize(double* Xi, double* Xi_std, int samples);
__device__ double covariance_norm(double* xi_std, double* xj_std, int samples);
__device__ double entropy(double* u, int samples);
__device__ double entropy_ij(double* ui, double* uj, double cov_ij, int samples);
__device__ double variance(double* X, int samples);
__device__ double diff_mutual_info(double* xi_std, double* xj_std, double cov_ij, int samples);
__device__ double entropy_cal_1(double y);
__device__ double entropy_cal_2(double y);


#endif