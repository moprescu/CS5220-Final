#include "main.cuh"

int main(int argc, char** argv){
    // Local variables
    gpu_lingam model;
    vector<double> X;
    int n, m;
    vector<int> causal_order;
    string dir;
    // Synthetic data descriptors
    vector<int> dims{100, 200, 500, 1000};
    vector<int> samples{1024, 2048, 4096, 8192};
    vector<string> graph_type{"S", "D", "SS"};
    // Real data descriptors
    vector<string> real_data{"e_coli_core", "pathfinder", "andes", "diabetes", "pigs", "link",
                                "iJR904", "munin", "iAF1260b", "iAF1260", "iY75_1357",
                                "iECDH10B_1368", "iML1515", "iEC1372_W3110"};

    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-i <filename>: set the input file name" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        return 0;
    }

    // Open Input File
    char* inname = find_string_option(argc, argv, "-i", nullptr);
    read_csv(inname, X, n, m);

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Move data to the GPU
    double* d_data; // Device pointer
    size_t size = X.size() * sizeof(double);
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, X.data(), size, cudaMemcpyHostToDevice);

    // Fit model
    auto start_time = std::chrono::steady_clock::now();
    causal_order = model.fit(d_data, n, m);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    std::cout << "Runtime: " << seconds << " seconds." << endl;
    // Save results to file
    if (fsave.good()) {
        for(int i = 0; i < causal_order.size()-1; i++){
            fsave << causal_order[i] << ", ";
        }
        fsave << causal_order[causal_order.size()-1] << endl;
    }
    cudaFree(d_data);
    return 0; 
}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

void read_csv(const std::string& filename, std::vector<double>& data, int& cols, int& rows) {
    // Data is stored column-wise, i.e. each line is a column in the matrix
    std::ifstream file(filename);
    std::string line;
    data.clear();
    rows = 0;
    cols = -1;  // Will use this to check consistency in column numbers

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    while (getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        int currentRowCols = 0;

        while (getline(lineStream, cell, ',')) {
            try {
                data.push_back(std::stod(cell)); // Convert string to double and store
                ++currentRowCols;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << " in cell: " << cell << std::endl;
                continue;
            }
        }

        // Check and set column count
        if (cols == -1) {
            cols = currentRowCols;
        } else if (cols != currentRowCols) {
            std::cerr << "Row " << rows + 1 << " has inconsistent number of columns." << std::endl;
        }

        ++rows;
    }
    file.close();
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

/*
#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << std::endl;
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        fsave << parts[i].x << " " << parts[i].y << std::endl;
    }

    fsave << std::endl;
}

// Particle Initialization
void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    int sx = (int)ceil(sqrt((double)num_parts));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < shuffle.size(); ++i) {
        shuffle[i] = i;
    }

    for (int i = 0; i < num_parts; ++i) {
        // Make sure particles are not spatially sorted
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];

        // Distribute particles evenly to ensure proper spacing
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);

        // Assign random velocities within a bound
        std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
    }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of particles" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Initialize Particles
    int num_parts = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    double size = sqrt(density * num_parts);

    particle_t* parts = new particle_t[num_parts];

    init_particles(parts, num_parts, size, part_seed);

    particle_t* parts_gpu;
    cudaMalloc((void**)&parts_gpu, num_parts * sizeof(particle_t));
    cudaMemcpy(parts_gpu, parts, num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    init_simulation(parts_gpu, num_parts, size);

    for (int step = 0; step < nsteps; ++step) {
        simulate_one_step(parts_gpu, num_parts, size);
        cudaDeviceSynchronize();

        // Save state if necessary
        if (fsave.good() && (step % savefreq) == 0) {
            cudaMemcpy(parts, parts_gpu, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save(fsave, parts, num_parts, size);
        }
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " << num_parts << " particles.\n";
    fsave.close();
    cudaFree(parts_gpu);
    delete[] parts;
}
*/