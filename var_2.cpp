#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>


constexpr auto t = 0.00001;
constexpr auto e_pow_2 = 0.00001 * 0.00001;
constexpr auto N = 15000;


void MatrixMultiply(const double *buf, int lines, const double *x, double *tmp) {
#pragma omp for
    for (int i = 0; i < lines; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += buf[i * N + j] * x[j];
        }
        tmp[i] = sum;
    }
}


double Norma(const double *vector, int size) {
    double norm_sum = 0.0;
#pragma omp parallel for reduction(+:norm_sum)
    for (int i = 0; i < size; i++) {
        norm_sum += vector[i] * vector[i];
    }
    return norm_sum;
}


void VectorDifference(const double *a_1, const double *a_2, int size, double *tmp) {
#pragma omp for
    for (int i = 0; i < size; i++) {
        tmp[i] = a_1[i] - a_2[i];
    }
}


void VectorMultiplyConst(const double *a_1, double value, int size, double *tmp) {
#pragma omp for
    for (int i = 0; i < size; i++) {
        tmp[i] = a_1[i] * value;
    }
}


bool NormaCompare(double norma, double norma_b, const std::vector<double> &x) {
    return (norma / norma_b < e_pow_2);
}


void Run() {
    std::vector<double> A((long long) N * (long long) N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }


    std::vector<double> b(N);
    std::vector<double> x(N);
    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
        x[i] = 0;
    }
    float *norm_sum = (float *) malloc(sizeof(float));
    *norm_sum = 0;

    double norma_b = Norma(b.data(), N);

    std::vector<double> tmp(N);
    std::vector<double> new_x_part(N);
    std::vector<double> multiply_tmp_const(N);

#pragma omp parallel
    {
        while (true) {
            MatrixMultiply(A.data(), N, x.data(), tmp.data());
#pragma omp barrier
            VectorDifference(tmp.data(), b.data(), N, tmp.data());
#pragma omp barrier

            norm_sum[0] = 0;

            double norma = Norma(tmp.data(), N);

            VectorMultiplyConst(tmp.data(), t, N, multiply_tmp_const.data());

#pragma omp barrier
            VectorDifference(x.data(), multiply_tmp_const.data(), N, x.data());

            if (NormaCompare(norma, norma_b, x)) {
                break;
            }
        }
    }
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();
    Run();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << elapsed.count() << " milliseconds\n";
    return 0;
}