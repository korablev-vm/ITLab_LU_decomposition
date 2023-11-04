#include <benchmark/benchmark.h>
#include "Matrix.h"

// Функция для заполнения матрицы случайными значениями
void FillMatrix(Matrix& matrix) {
    for (int i = 0; i < matrix.get_size(); ++i) {
        for (int j = 0; j < matrix.get_size(); ++j) {
            matrix[i][j] = rand() % 100; // Заполнение случайными числами от 0 до 99
        }
    }
}

// Функция бенчмарка для LU-разложения
static void BM_LU_Decomposition(benchmark::State& state) {
    int n = state.range(0);
    Matrix matrix(n);
    FillMatrix(matrix); // Заполнение матрицы перед замером времени

    for (auto _ : state) {
        matrix.LU_Decomposition(); // Выполнение LU-разложения
    }
}

// Регистрация функции бенчмарка
BENCHMARK(BM_LU_Decomposition)->Range(8, 8 << 10); // Тестируем матрицы размером от 8x8 до 1024x1024

BENCHMARK_MAIN(); // Точка входа для бенчмарка
