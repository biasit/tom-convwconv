#include <iostream>
#include <chrono>

#include "lut_amm.hpp"

using namespace std;
using namespace std::chrono;

int main(void){
    int n = 100000;
    int d = 125;
    int m = 125;
    int ncode = 4;
    MithralMatmul matmul(n,d,m,ncode, -1.0);
    cout << "Starting test" << endl;
    cout << endl;
    auto start = high_resolution_clock::now();

    matmul.lut();
    Eigen::MatrixXf out = Eigen::MatrixXf::Zero(n, m);
    for (int i = 0; i < 1000; i++) {
        matmul.run_matmul(false);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Timing of LUT Matmul: " << duration.count() << "ms" << endl << endl;


    start = high_resolution_clock::now();
    
    for (int i = 0; i < 1000; i++) {
        out = matmul.X * matmul.Q;
    }
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    cout << "Timing of Normal Matmul: " << duration.count() << "ms" << endl;
}