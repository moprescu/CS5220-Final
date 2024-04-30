#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include "gpu_lingam.cuh"

using namespace std;

void read_csv(const string& filename, vector<double>& data, int& rows, int& cols);
int find_arg_idx(int argc, char** argv, const char* option);
int find_int_arg(int argc, char** argv, const char* option, int default_value);
char* find_string_option(int argc, char** argv, const char* option, char* default_value);
int main(int argc, char** argv);