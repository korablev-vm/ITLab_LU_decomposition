#include "Matrix.h"
#include <iostream>
#include <chrono>

int main() {
    size_t n;
    std::cout << "Enter matrix size: ";
    std::cin >> n;

    Matrix matrix(n);

    // ���������� �������: ������������ �������� = 1, ��������� = 0
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }

    // �������� ������ L � U ��� LU-����������
    Matrix L(n), U(n);

    // ��������� ������� LU-����������
    auto start = std::chrono::high_resolution_clock::now();
    matrix.LU_Decomposition(L, U);
    auto end = std::chrono::high_resolution_clock::now();

    // ����� ������� ����������
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time for LU decomposition: " << diff.count() << " seconds\n";

    return 0;
}
