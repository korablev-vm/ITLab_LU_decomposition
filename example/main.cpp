#include "Matrix.h"
#include <iostream>
#include <chrono>
#include <mkl.h>

using namespace std;

/*
доделать:
advisor подключить (почитать, что это такое). inspector - ищем ошибки работы с памятью/гонки данных; интринсики. advisor - тул, чтобы не имея конкретного железа можно было
предсказать, насколько хорошо программа распараллелится. дает свои рекомендации. к следующему собранию принести рофлайны (результаты работы этих инструментов). Рофлайн модель - способ визуализировать текущие показатели
производительности
*/

int main() {
    size_t n;
    cout << "Enter matrix size: ";
    cin >> n;

    Matrix matrix(n);

    // Заполнение матрицы: диагональные элементы = 1, остальные = 0
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }

    Matrix L(n), U(n);
    Matrix clear = matrix;
    auto start = chrono::high_resolution_clock::now();
    matrix.LU_Decomposition(L, U);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << "Time for LU decomposition: " << diff.count() << " seconds\n";
    //cout << L << endl;
    //cout << U << endl;

    //// MKL
    //int* ipiv = new int[n];
    //auto start_mkl = chrono::high_resolution_clock::now();
    //LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, clear.get_data(), n, ipiv);
    //auto end_mkl = chrono::high_resolution_clock::now();
    //chrono::duration<double> diff_mkl = end_mkl - start_mkl;
    //cout << "Time for LU decomposition (MKL): " << diff_mkl.count() << " seconds\n";

    //for (int v = 0; v < 7; v++)
    //{
    //    start_mkl = chrono::high_resolution_clock::now();
    //    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, clear.get_data(), n, ipiv);
    //    end_mkl = chrono::high_resolution_clock::now();
    //    diff_mkl = end_mkl - start_mkl;
    //    cout << "Time for LU decomposition (MKL): " << diff_mkl.count() << " seconds\n";
    //}

    return 0;
}
