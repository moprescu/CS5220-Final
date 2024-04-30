#include "main.hpp"

int main(int argc, char** argv){
    direct_lingam model;
    vector<vector<data_type>> X;
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
    X = read_csv(inname);

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Fit model
    auto start_time = std::chrono::steady_clock::now();
    causal_order = model.fit(X);
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
    return 0; 
}

vector<vector<data_type>> read_csv(string dir){
    ifstream file(dir);
    string line;
    vector<vector<data_type>> data;
    vector<data_type> row;
    getline(file, line);
    while(line != ""){
        vector<data_type> ().swap(row);
        while( line.find(',') != string::npos){
            row.push_back(stod(line.substr(0, line.find(','))));
            line.erase(0, line.find(',') + 1);
        }
        row.push_back(stod(line));
        getline(file, line);
        data.push_back(row);
    }
    return data;
}

bool vector_check(vector<vector<data_type>> data){
    vector<data_type> temp = *data.begin();
    int size = temp.size();
    for (auto i = data.begin(); i!= data.end(); ++i){
        temp = *i;
        if (temp.size() != size)
            return false;
    }
    return true;
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