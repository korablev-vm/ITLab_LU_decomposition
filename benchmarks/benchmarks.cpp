#include <benchmark/benchmark.h>
#include "Matrix.h"
#include <cstdlib>

void FillMatrix(Matrix& matrix) {
    for (int i = 0; i < matrix.get_size(); ++i) {
        for (int j = 0; j < matrix.get_size(); ++j) {
            if (i == j) {
                matrix(i, j) = i;
            }
            else {
                matrix(i, j) = 0;
            }
        }
    }
}

static void BM_LU_Decomposition(benchmark::State& state) {
    int n = state.range(0);
    Matrix matrix(n);
    FillMatrix(matrix);

    for (auto _ : state) {
        Matrix L(n), U(n);
        matrix.LU_Decomposition(L, U);
    }
}

BENCHMARK(BM_LU_Decomposition)->RangeMultiplier(2)->Range(128, 4096);

BENCHMARK_MAIN();
