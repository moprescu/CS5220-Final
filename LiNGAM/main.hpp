#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include "direct_lingam.hpp"


using namespace std;

vector<vector<data_type>> read_csv(string dir);
bool vector_check(vector<vector<data_type>> data);
int find_arg_idx(int argc, char** argv, const char* option);
int find_int_arg(int argc, char** argv, const char* option, int default_value);
char* find_string_option(int argc, char** argv, const char* option, char* default_value);
int main(int argc, char** argv);