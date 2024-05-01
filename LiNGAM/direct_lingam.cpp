#include "direct_lingam.hpp"

direct_lingam::direct_lingam(/* args */){
}

direct_lingam::~direct_lingam(){
}


vector<int> direct_lingam::fit(vector<vector<data_type>> X){
    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    causal_order.clear();
    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);

    for (int dim = 0; dim < dims; dim++){
        int root = search_causal_order(X, U);
        for(auto i = U.begin(); i != U.end(); i++){
            if( *i != root){
                if(base){
                    X[*i] = residual_base(X[*i], X[root]);
                }
                else{
                    X[*i] = residual(X[*i], X[root]);
                }
            }
        }
        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        causal_order.push_back(root);
    }
    if (verbose){
        cout << "causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}

vector<int> direct_lingam::fit_opt(vector<vector<data_type>> X){
    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    causal_order.clear();
    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);

    for (int dim = 0; dim < dims; dim++){
        int root = search_causal_order_opt(X, U);
        for(auto i = U.begin(); i != U.end(); i++){
            if( *i != root){
                if(base){
                    X[*i] = residual_base(X[*i], X[root]);
                }
                else{
                    X[*i] = residual(X[*i], X[root]);
                }
            }
        }
        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        causal_order.push_back(root);
    }
    if (verbose){
        cout << "causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}

vector<int> direct_lingam::get_causal_order(){
    return causal_order;
}

vector<data_type> direct_lingam::residual(vector<data_type> xi, vector<data_type> xj){
    vector<data_type> result;
    data_type cov = covariance(xi, xj);
    data_type var = variance(xj);

    if (base){
        data_type temp = pow((1. - pow(cov,2)), 0.5);
        for(int i = 0; i < xi.size(); i++){
            result.push_back((xi[i] - cov * xj[i])/temp);
        }
    }
    else {
        for(int i = 0; i < xi.size(); i++){
            result.push_back(xi[i] - cov/var * xj[i]);
        }
    }
    return result;
}

void direct_lingam::residual_from_norm(vector<data_type> xi, vector<data_type> xj, vector<data_type> &ri_j, vector<data_type> &rj_i){
    vector<data_type> result;
    data_type cov = 0;
    for(int i = 0; i < xi.size(); i++){
        cov += xi[i] * xj[i];
    }
    cov /= (data_type)(xi.size()-1);
    for(int i = 0; i < xi.size(); i++){
        ri_j[i] = xi[i] - cov * xj[i];
        rj_i[i] = xj[i] - cov * xi[i];
    }
}

vector<data_type> direct_lingam::residual_base(vector<data_type> xi, vector<data_type> xj){
    vector<data_type> result;
    data_type cov = covariance(xi, xj);
    data_type var = variance(xj);

    for(int i = 0; i < xi.size(); i++){
        result.push_back(xi[i] - cov * xj[i]);
    }
    return result;
}

data_type direct_lingam::entropy(vector<data_type> u){
    data_type k1 = 79.047;
    data_type k2 = 7.4129;
    data_type gamma = 0.37457;
    data_type result = 0;
    data_type cal_1;
    data_type cal_2;
    data_type _size = data_type(u.size());
    cal_1 = accumulate(u.begin(), u.end(), 0.0, entropy_cal_1)/_size;
    cal_2 = accumulate(u.begin(), u.end(), 0.0, entropy_cal_2)/_size;
    return (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
}

data_type direct_lingam::entropy_ij(vector<data_type> ui, vector<data_type> uj, data_type cov_ij){
    data_type k1 = 79.047;
    data_type k2 = 7.4129;
    data_type gamma = 0.37457;
    data_type cal_1 = 0;
    data_type cal_2 = 0;
    data_type norm_cov = pow(1-pow(cov_ij, 2), 0.5);
    for (int i = 0; i < ui.size(); i++){
        cal_1 += entropy_cal_1(0, (ui[i]-uj[i]*cov_ij)/norm_cov);
        cal_2 += entropy_cal_2(0, (ui[i]-uj[i]*cov_ij)/norm_cov);
    }
    cal_1 /= data_type(ui.size());
    cal_2 /= data_type(ui.size());
    return (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
}

data_type direct_lingam::diff_mutual_info(vector<data_type> xi_std, vector<data_type> xj_std, 
                                            vector<data_type> ri_j, vector<data_type> rj_i){

    data_type std_ri_j = pow(variance(ri_j), 0.5);
    data_type std_rj_i = pow(variance(rj_i), 0.5);
    if (base){
        std_ri_j = 1.0;
        std_rj_i = 1.0;
    }
    int _size = xi_std.size();
    for(int i = 0; i < _size; i++){
        ri_j[i] = ri_j[i] / std_ri_j;
        rj_i[i] = rj_i[i] / std_rj_i;
    }
    return (entropy(xj_std) + entropy(ri_j)) - (entropy(xi_std) + entropy(rj_i));
}

data_type direct_lingam::diff_mutual_info_X_entropy(data_type xi_entropy, data_type xj_entropy,
                                            vector<data_type> xi_std, vector<data_type> xj_std, data_type cov_ij){
    return (xj_entropy + entropy_ij(xi_std, xj_std, cov_ij)) - (xi_entropy + entropy_ij(xj_std, xi_std, cov_ij));
}


vector<vector<int>> direct_lingam::search_candidate(vector<int> U){
    vector<vector<int>> result;
    vector<int> empty_vector;
    result.push_back(U);
    result.push_back(empty_vector);
    return result;
}

int direct_lingam::search_causal_order(vector<vector<data_type>> &X, vector<int> U){
    vector<int> Uc;
    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];

    if(Uc.size() == 1){
        return Uc[0];
    }

    vector<data_type> M_list;
    vector<data_type> xi_std;
    vector<data_type> xj_std;
    vector<data_type> ri_j;
    vector<data_type> rj_i;

    int temp = 0;

    for(auto i = Uc.begin(); i != Uc.end(); i++){
        data_type M = 0;
        for (auto j = U.begin(); j != U.end(); j++){
            if (*i != *j){
                xi_std = normalize(X[*i]);
                xj_std = normalize(X[*j]);
                ri_j = residual(xi_std, xj_std);
                rj_i = residual(xj_std,  xi_std);
                M += pow(std::min(data_type(0), diff_mutual_info(xi_std, xj_std, ri_j, rj_i)), 2);
            }
        }
        M_list.push_back( -1.0 * M);
    }
    return Uc[std::max_element(M_list.begin(), M_list.end()) - M_list.begin()];
}

int direct_lingam::search_causal_order_opt(vector<vector<data_type>> &X, vector<int> U){
    vector<int> Uc;
    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];

    if(Uc.size() == 1){
        return Uc[0];
    }

    vector<vector<data_type>> X_norm(X.size());
    vector<data_type> X_entropy(X.size());
    vector<data_type> M_list;
    vector<data_type> xi_std;
    vector<data_type> xj_std;
    vector<data_type> ri_j(X[0].size()), rj_i(X[0].size());
    data_type running_minimum = DBL_MAX;
    int minimum_idx = -1;

    //Normalize data once
    for(int i = 0; i<X.size(); i++){
        X_norm[i] = normalize(X[i]);
        X_entropy[i] = entropy(X_norm[i]);
    }

    for(auto i = Uc.begin(); i != Uc.end(); i++){
        data_type M = 0;
        for (auto j = U.begin(); j != U.end(); j++){
            if (*i != *j){
                double cov_ij = covariance_norm(X_norm[*i], X_norm[*j]);
                M += pow(std::min(data_type(0), diff_mutual_info_X_entropy(X_entropy[*i], X_entropy[*j], X_norm[*i], X_norm[*j], cov_ij)), 2);
                //residual_from_norm(X_norm[*i], X_norm[*j], ri_j, rj_i);
                //M += pow(std::min(data_type(0), diff_mutual_info_X_entropy(X_entropy[*i], X_entropy[*j], ri_j, rj_i)), 2);
            }
            if (M > running_minimum){
                break;
            }
        }
        if (M < running_minimum){
            running_minimum = M;
            minimum_idx = *i;
        }
    }
    return minimum_idx;
}

vector<data_type> normalize(vector<data_type> X){
    data_type mean = accumulate(X.begin(), X.end(), 0.0)/ data_type(X.size());
    data_type var = 0;
    for(auto i = X.begin(); i != X.end(); i++){
        var += pow((*i - mean), 2);
    }
    data_type std_cal = sqrt(var/data_type(X.size() - 1));
    for(auto i = X.begin(); i != X.end(); i++){
        *i = (*i - mean) / std_cal;
    }
    return X;
}

void vector_print(vector<data_type> data){    
    for(auto i = data.begin(); i != data.end(); i++){
        std::cout << *i << ' ';
    }
    std::cout << endl;
}

data_type variance(vector<data_type> X){
    data_type mean = accumulate(X.begin(), X.end(), 0.0)/ data_type(X.size());
    data_type var = 0;
    for(auto i = X.begin(); i != X.end(); i++){
        var += pow((*i - mean), 2);
    }
    return var/(data_type(X.size()-1));
}

data_type direct_lingam::covariance(vector<data_type> X, vector<data_type> Y){
    int n = X.size();
    data_type X_mean = accumulate(X.begin(), X.end(), 0.0)/ data_type(n);
    data_type Y_mean = accumulate(Y.begin(), Y.end(), 0.0)/ data_type(n);
    data_type sum = 0;
    for(int i = 0; i < n; i++){
        sum += (X[i] - X_mean) * (Y[i] - Y_mean);
    }
    data_type result = sum/data_type(n-1);
    if(base){
        if(result > 0.99){
            result = 0.99;
        }
        else if(result < -0.99){
            result = -0.99;
        }
    }
    return result;
}

data_type covariance_norm(vector<data_type> X, vector<data_type> Y){
    data_type cov = 0;
    for(int i = 0; i < X.size(); i++){
        cov += X[i] * Y[i];
    }
    cov /= (data_type)(X.size() - 1);
    return cov;
}


data_type entropy_cal_1(data_type x, data_type y){
    return x + log(cosh(y));
}

data_type entropy_cal_2(data_type x, data_type y){
    return x + y * (exp(-0.5* pow(y, 2)));
}

void direct_lingam::set_base_mode(bool mode){
    base = mode;
}

void direct_lingam::set_verbose_mode(bool mode){
    verbose = mode;
}