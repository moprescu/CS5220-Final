#include "gpu_lingam.cuh"

gpu_lingam::gpu_lingam(/* args */){
}

gpu_lingam::~gpu_lingam(){
}

vector<int> gpu_lingam::fit(double* d_X, int n, int m){
    dims = m;
    samples = n;
    causal_order.clear();
    causal_order.resize(dims);
    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);
    int blks;
    
    // Define device variables and move to GPU
    int* d_causal_order;
    int* d_U;
    double* d_M_list;
    double* d_X_norm;
    //double* cov_X;
    cudaMalloc(&d_causal_order, dims * sizeof(int));
    cudaMalloc(&d_U, dims * sizeof(int));
    cudaMemcpy(d_U, U.data(), dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_M_list, dims * sizeof(double));
    cudaMalloc(&d_X_norm, n * m * sizeof(double));

    // This part needs to be done in sequence
    for (int dim = 0; dim < dims; dim++){
        // Set d_M_list
        cudaMemset(d_M_list, 0, dims * sizeof(double));
        // Call GPU kernels, all computations are done on the GPU
        // GPU kernel: search for the root node, add to causal order and remove from u
        blks = (dims + NUM_THREADS - 1) / NUM_THREADS;
        normalize_data<<<blks, NUM_THREADS>>>(d_X, d_X_norm, samples, dims);
        cudaDeviceSynchronize();

        /*
        // Initial GPU implementation
        blks = (pow((dims - dim), 2) + NUM_THREADS - 1) / NUM_THREADS;
        compute_M<<<blks, NUM_THREADS>>>(d_X_norm, d_M_list, d_U, dims, samples, dim);
        cudaDeviceSynchronize();
        */

        /*
        // Parallel reduction
        int pr_NUM_THREADS = 128;
        int pr_blks = (pow((dims - dim), 2) + 1) / 2;
        size_t sharedSize = (2*pr_NUM_THREADS + 14) * sizeof(double);
        block_compute_M<<<pr_blks, pr_NUM_THREADS, sharedSize>>>(d_X_norm, d_M_list, d_U, dims, samples, dim, pr_NUM_THREADS/2);
        cudaDeviceSynchronize();
        */


        // Hybrid implementation
        if (dims-dim > 500){
            blks = (pow((dims - dim), 2) + NUM_THREADS - 1) / NUM_THREADS;
            compute_M<<<blks, NUM_THREADS>>>(d_X_norm, d_M_list, d_U, dims, samples, dim);
            cudaDeviceSynchronize();
        }
        else{
            // Parallel reduction
            int pr_NUM_THREADS = 128;
            int pr_blks = (pow((dims - dim), 2) + 1) / 2;
            size_t sharedSize = (2*pr_NUM_THREADS + 14) * sizeof(double);
            block_compute_M<<<pr_blks, pr_NUM_THREADS, sharedSize>>>(d_X_norm, d_M_list, d_U, dims, samples, dim, pr_NUM_THREADS/2);
            cudaDeviceSynchronize();
        }

        
        update_order<<<1, 1>>>(d_U, d_causal_order, d_M_list, dims, dim);
        cudaDeviceSynchronize(); 

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
        // GPU kernel: residualize data
        blks = (dims + NUM_THREADS - 1) / NUM_THREADS;
        residualize_data<<<blks, NUM_THREADS>>>(d_X, d_causal_order, dims, samples, dim);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(causal_order.data(), d_causal_order, dims * sizeof(int), cudaMemcpyDeviceToHost);
    if (verbose){
        cout << "causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}

vector<int> gpu_lingam::get_causal_order(){
    return causal_order;
}

__global__ void normalize_data(double* d_X, double* d_X_norm, int samples, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dims) {
        normalize(d_X + tid * samples, d_X_norm + tid * samples, samples);
    }
}

__global__ void compute_M(double* d_X_norm, double* d_M_list, int* d_U, int dims, int samples, int iter){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = pow((dims - iter), 2);
    if (tid < total_threads) {
        int i = d_U[tid / (dims-iter)];
        int j = d_U[tid % (dims-iter)];
        if (i < j) {
            double cov_ij = covariance_norm(d_X_norm + i * samples, d_X_norm + j * samples, samples);
            double diff_mi = diff_mutual_info(d_X_norm + i * samples, d_X_norm + j * samples, cov_ij, samples);
            double M = -1.0 * pow(min(double(0), diff_mi), 2);
            atomicAdd(&d_M_list[tid / (dims-iter)], M);
            M = -1.0 * pow(min(double(0), -diff_mi), 2);
            atomicAdd(&d_M_list[tid % (dims-iter)], M);
        }
    }
}

__global__ void block_compute_M(double* d_X_norm, double* d_M_list, int* d_U, int dims, int samples, int iter, int num_threads){
    extern __shared__ double sharedMem[]; // Data for loading the columns to be worked on & intermediate calculations
    double* cov = sharedMem;
    double* calcs = sharedMem + 2;
    double* calc1 = sharedMem + 14;
    double* calc2 = sharedMem + 2* num_threads + 14;
    double k1 = 79.047;
    double k2 = 7.4129;
    double gamma = 0.37457;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int q = tid / num_threads;
    if (2*bid + q < (dims-iter)*(dims-iter)){
        int i = d_U[(2*bid + q) / (dims-iter)];
        int j = d_U[(2*bid + q) % (dims-iter)];
        double* col_i = d_X_norm + i * samples;
        double* col_j = d_X_norm + j * samples;
    
        if (i !=  j){
            //Calulate covariance
            calc1[tid] = 0.0;
            for(int k = tid - q*num_threads; k < samples; k+=num_threads){
                if (k < samples){
                    calc1[tid] += col_i[k] * col_j[k];
                }
            }
            __syncthreads(); // Ensure data is loaded and cov entries are calculated
            // Calculate covariance
            if (tid % num_threads == 0){
                *(cov + q) = 0.0;
                for(int k = 0; k < num_threads; k++){
                    *(cov + q) += calc1[k + q*num_threads];
                }
                *(cov + q) /= (double)(samples - 1);
            }
            __syncthreads(); // Covariance is calculated
            // Calculate entropy entries
            // Initialize entropy entries
            calc1[tid] = 0.0; // j entropy cal_1
            calc2[tid] = 0.0; // j entropy cal_2
            double norm_cov = pow(1-pow(*(cov+q), 2), 0.5);
            for(int k = tid - q*num_threads; k < samples; k+=num_threads){
                if (k < samples){
                    calc1[tid] += entropy_cal_1(col_j[k]);
                    calc2[tid] += entropy_cal_2(col_j[k]);
                }
            }
            __syncthreads(); // Entropy elements are calculated
            if (tid % num_threads == 0){
                // Coalesce calcs and calculate:
                //(1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                *(calcs + 6 * q) = 0.0;
                *(calcs + 6 * q + 1) = 0.0;
                for(int k = 0; k < num_threads; k++){
                    *(calcs + 6 * q) += calc1[k + q*num_threads];
                    *(calcs + 6 * q + 1) += calc2[k + q*num_threads];
                }
                *(calcs + 6 * q) /= double(samples);
                *(calcs + 6 * q + 1) /= double(samples);
                *(calcs + 6 * q + 2) = (1 + log(2 * M_PI)) / 2 - k1 * pow(*(calcs + 6 * q) - gamma, 2) - k2 * pow(*(calcs + 6 * q + 1), 2);
            }
            __syncthreads();

            ////// Another block /////
            calc1[tid] = 0.0; // ij entropy cal_1
            calc2[tid] = 0.0; // ij entropy cal_2
            for(int k = tid - q*num_threads; k < samples; k+=num_threads){
                if (k < samples){
                    calc1[tid] += entropy_cal_1((col_i[k]-col_j[k]*(*(cov+q)))/norm_cov);
                    calc2[tid] += entropy_cal_2((col_i[k]-col_j[k]*(*(cov+q)))/norm_cov);
                }
            }
            __syncthreads();
            if (tid % num_threads == 0){
                // Coalesce calcs and calculate:
                //(1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                calcs[0] = 0.0;
                calcs[1] = 0.0;
                for(int k = 0; k < num_threads; k++){
                    *(calcs + 6 * q) += calc1[k + q*num_threads];
                    *(calcs + 6 * q + 1) += calc2[k + q*num_threads];
                }
                *(calcs + 6 * q) /= double(samples);
                *(calcs + 6 * q + 1) /= double(samples);
                *(calcs + 6 * q + 3) = (1 + log(2 * M_PI)) / 2 - k1 * pow(*(calcs + 6 * q) - gamma, 2) - k2 * pow(*(calcs + 6 * q + 1), 2);
            }
            __syncthreads();

            ////// Another block /////
            calc1[tid] = 0.0; // ij entropy cal_1
            calc2[tid] = 0.0; // ij entropy cal_2
            for(int k = tid - q*num_threads; k < samples; k+=num_threads){
                if (k < samples){
                    calc1[tid] += entropy_cal_1((col_j[k]-col_i[k]*(*(cov+q)))/norm_cov);
                    calc2[tid] += entropy_cal_2((col_j[k]-col_i[k]*(*(cov+q)))/norm_cov);
                }
            }
            __syncthreads();
            if (tid % num_threads == 0){
                // Coalesce calcs and calculate:
                //(1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                calcs[0] = 0.0;
                calcs[1] = 0.0;
                for(int k = 0; k < num_threads; k++){
                    *(calcs + 6 * q) += calc1[k + q*num_threads];
                    *(calcs + 6 * q + 1) += calc2[k + q*num_threads];
                }
                *(calcs + 6 * q) /= double(samples);
                *(calcs + 6 * q + 1) /= double(samples);
                *(calcs + 6 * q + 4) = (1 + log(2 * M_PI)) / 2 - k1 * pow(*(calcs + 6 * q) - gamma, 2) - k2 * pow(*(calcs + 6 * q + 1), 2);
            }
            __syncthreads(); 

            ////// Another block /////
            calc1[tid] = 0.0; // ij entropy cal_1
            calc2[tid] = 0.0; // ij entropy cal_2
             for(int k = tid - q*num_threads; k < samples; k+=num_threads){
                if (k < samples){
                    calc1[tid] += entropy_cal_1(col_i[k]);
                    calc2[tid] += entropy_cal_2(col_i[k]);
                }
            }
            __syncthreads();
            if (tid % num_threads == 0){
                // Coalesce calcs and calculate:
                //(1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                calcs[0] = 0.0;
                calcs[1] = 0.0;
                for(int k = 0; k < num_threads; k++){
                    *(calcs + 6 * q) += calc1[k + q*num_threads];
                    *(calcs + 6 * q + 1) += calc2[k + q*num_threads];
                }
                *(calcs + 6 * q) /= double(samples);
                *(calcs + 6 * q + 1) /= double(samples);
                *(calcs + 6 * q + 5) = (1 + log(2 * M_PI)) / 2 - k1 * pow(*(calcs + 6 * q) - gamma, 2) - k2 * pow(*(calcs + 6 * q + 1), 2);
                //// Add to global M
                double M = -1.0 * pow(min(double(0), *(calcs + 6 * q + 2) + *(calcs + 6 * q + 3) - *(calcs + 6 * q + 4) - *(calcs + 6 * q + 5)), 2);
                atomicAdd(&d_M_list[(2*bid+q) / (dims-iter)], M);
            }
        }
    }
}

__global__ void update_order(int* d_U, int* d_causal_order, double* d_M_list, int dims, int iter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int root = d_U[max_index(d_M_list, dims - iter)];
        remove_element_by_value(d_U, root, dims - iter);
        d_causal_order[iter] = root;
    }
}

__global__ void residualize_data(double* d_X, int* d_causal_order, int dims, int samples, int iter){
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int root = d_causal_order[iter];
    if((j >= dims) || j == root){
        return;
    }
    double* X_root = d_X + root * samples;
    double* Xj = d_X + j * samples;

    // Calculate mean and variance of root
    double sum_root = 0.0;
    double sum_root_squared = 0.0;
    double sum_j = 0.0;
    double sum_root_j = 0.0;
    for (int i = 0; i < samples; i++) {
        sum_root += X_root[i];
        sum_root_squared += X_root[i] * X_root[i];
        sum_j += Xj[i];
        sum_root_j += X_root[i] * Xj[i];
    }
    sum_root /= (double)samples;
    sum_j /= (double)samples;
    double beta = (sum_root_j - sum_j * sum_root * (double)samples) / (sum_root_squared - sum_root * sum_root * (double)samples);

    // Apply the residualization
    for (int i = 0; i < samples; i++) {
        Xj[i] = Xj[i] - beta * X_root[i];
    }
}

__device__ void remove_element_by_value(int* array, int value, int size) {
    int index = -1;
    // Find the index of the element with value
    for (int i = 0; i < size; ++i) {
        if (array[i] == value) {
            index = i;
            break;
        }
    }
    // Shift elements
    if (index != -1) {
        for (int i = index; i < size - 1; ++i) {
            array[i] = array[i + 1];
        }
        // Set the last element to -1 to indicate removal
        array[size - 1] = -1;
    }
}

__device__ int max_index(double* array, int size){
    int max_index = -1;
    double max_element = -DBL_MAX;
    for (int i = 0; i < size; i++){
        if (array[i] > max_element){
            max_index = i;
            max_element = array[i];
        }
    }
    return max_index;
}

__device__ void normalize(double* Xi, double* Xi_std, int samples){
    double sum = 0, sum_squares = 0;
    for (int i = 0; i < samples; ++i) {
        sum += Xi[i];
        sum_squares += pow(Xi[i], 2);
    }
    double mean = sum / samples;
    double sd = sqrt((sum_squares - samples * pow(mean, 2)) / (samples - 1));
    for (int i = 0; i < samples; ++i) {
        Xi_std[i] = (Xi[i] - mean) / sd;
    }
}

__device__ double covariance_norm(double* xi_std, double* xj_std, int samples){
    double cov = 0;
    for(int i = 0; i < samples; i++){
        cov += xi_std[i] * xj_std[i];
    }
    cov /= (double)(samples - 1);
    return cov;
}


__device__ double entropy(double* u, int samples){
    double k1 = 79.047;
    double k2 = 7.4129;
    double gamma = 0.37457;
    double cal_1 = 0;
    double cal_2 = 0;
    for (int i = 0; i < samples; i++){
        cal_1 += entropy_cal_1(u[i]);
        cal_2 += entropy_cal_2(u[i]);
    }
    cal_1 /= double(samples);
    cal_2 /= double(samples);
    return (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
}

__device__ double entropy_ij(double* ui, double* uj, double cov_ij, int samples){
    double k1 = 79.047;
    double k2 = 7.4129;
    double gamma = 0.37457;
    double cal_1 = 0;
    double cal_2 = 0;
    double norm_cov = pow(1-pow(cov_ij, 2), 0.5);
    for (int i = 0; i < samples; i++){
        cal_1 += entropy_cal_1((ui[i]-uj[i]*cov_ij)/norm_cov);
        cal_2 += entropy_cal_2((ui[i]-uj[i]*cov_ij)/norm_cov);
    }
    cal_1 /= double(samples);
    cal_2 /= double(samples);
    return (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
}

__device__ double diff_mutual_info(double* xi_std, double* xj_std, double cov_ij, int samples){
    return (entropy(xj_std, samples) + entropy_ij(xi_std, xj_std, cov_ij, samples)) - (entropy(xi_std, samples) + entropy_ij(xj_std, xi_std, cov_ij, samples));
}

__device__ double entropy_cal_1(double y){
    return log(cosh(y));
}

__device__ double entropy_cal_2(double y){
    return y * (exp(-0.5* pow(y, 2)));
}