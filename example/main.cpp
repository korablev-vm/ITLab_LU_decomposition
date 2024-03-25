#include "Matrix.h"
#include <iostream>
#include <chrono>
#include <mkl.h>

using namespace std;

int main() {
    size_t n = 1024;

    Matrix<double> matrix = Matrix<double>::createDiagonallyDominantMatrix(n);

    Matrix<double> L(n), U(n);
    Matrix<double> clear = matrix;
    auto start = std::chrono::high_resolution_clock::now();
    matrix.LU_Decomposition(L, U);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time for LU decomposition: " << diff.count() << " seconds\n";

    /* start = std::chrono::high_resolution_clock::now();
     clear.LU_Decomposition_base(L, U);
     end = std::chrono::high_resolution_clock::now();
     diff = end - start;
     std::cout << "Time for LU decomposition (base): " << diff.count() << " seconds\n";*/

     //// MKL
    int* ipiv = new int[n];
    auto start_mkl = std::chrono::high_resolution_clock::now();
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, clear.get_data(), n, ipiv);
    auto end_mkl = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_mkl = end_mkl - start_mkl;
    std::cout << "Time for LU decomposition (MKL): " << diff_mkl.count() << " seconds\n";

    delete[] ipiv;
    return 0;
}
