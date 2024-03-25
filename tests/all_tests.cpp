#include <gtest/gtest.h>
#include "Matrix.h"
#include <cmath>
#include <cstdlib>
#include <gmock/gmock.h>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Функция для проверки равенства двух матриц
bool AreMatricesEqual(const Matrix<double>& m1, const Matrix<double>& m2) {
    if (m1.get_size() != m2.get_size()) return false;
    for (size_t i = 0; i < m1.get_size(); ++i) {
        for (size_t j = 0; j < m1.get_size(); ++j) {
            if (std::abs(m1(i, j) - m2(i, j)) > 1e-6) { // Проверка с некоторой погрешностью
                return false;
            }
        }
    }
    return true;
}

// Тест на корректность LU-разложения
TEST(Matrix, LUDecompositionCorrectness) {
    Matrix<double> m(3);
    m(0, 0) = 2;
    m(0, 1) = -1;
    m(0, 2) = -2;
    m(1, 0) = -4;
    m(1, 1) = 6;
    m(1, 2) = 3;
    m(2, 0) = -4;
    m(2, 1) = -2;
    m(2, 2) = 8;

    Matrix<double> L(3), U(3);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix<double> product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}

// Тест на LU-разложение с единичной матрицей
TEST(Matrix, LUDecompositionIdentityMatrix) {
    Matrix<double> m(3);
    m(0, 0) = 1;
    m(0, 1) = 0;
    m(0, 2) = 0;
    m(1, 0) = 0;
    m(1, 1) = 1;
    m(1, 2) = 0;
    m(2, 0) = 0;
    m(2, 1) = 0;
    m(2, 2) = 1;

    Matrix<double> L(3), U(3);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    // L и U должны быть единичными матрицами
    EXPECT_TRUE(AreMatricesEqual(m, L));
    EXPECT_TRUE(AreMatricesEqual(m, U));
}

// Тест на LU-разложение с матрицей, содержащей отрицательные элементы
TEST(Matrix, LUDecompositionWithNegativeElements) {
    Matrix<double> m(2);
    m(0, 0) = -1;
    m(0, 1) = -2;
    m(1, 0) = -3;
    m(1, 1) = -4;

    Matrix<double> L(2), U(2);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix<double> product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}

// Тест на LU-разложение с матрицей, содержащей дробные числа
TEST(Matrix, LUDecompositionWithFractionalElements) {
    Matrix<double> m(2);
    m(0, 0) = 0.5;
    m(0, 1) = 1.5;
    m(1, 0) = 1.2;
    m(1, 1) = 2.3;

    Matrix<double> L(2), U(2);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix<double> product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}

// Тест на LU-разложение с матрицей большого размера
TEST(Matrix, LUDecompositionLargeMatrix) {
    const size_t size = 1024; // Используем матрицу размером 10x10
    Matrix<double> m = Matrix<double>::createDiagonallyDominantMatrix(size);

    Matrix<double> L(size), U(size);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix<double> product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}

// Тест на корректность умножения матриц
TEST(Matrix, CanMultiplyMatrices) {
    Matrix<double> m1(2), m2(2);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(1, 0) = 3;
    m1(1, 1) = 4;

    m2(0, 0) = 5;
    m2(0, 1) = 6;
    m2(1, 0) = 7;
    m2(1, 1) = 8;

    Matrix<double> result = m1.multiply(m2);
    EXPECT_DOUBLE_EQ(result(0, 0), 19);
    EXPECT_DOUBLE_EQ(result(0, 1), 22);
    EXPECT_DOUBLE_EQ(result(1, 0), 43);
    EXPECT_DOUBLE_EQ(result(1, 1), 50);
}

// Тест на выброс исключения при умножении матриц разного размера
TEST(Matrix, ThrowsOnMultiplyDifferentSizes) {
    Matrix<double> m1(2), m2(3);
    EXPECT_THROW(m1.multiply(m2), std::invalid_argument);
}

// Тест на корректность копирования матрицы
TEST(Matrix, CanCopyMatrix) {
    Matrix<double> m1(2);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(1, 0) = 3;
    m1(1, 1) = 4;
    Matrix<double> m2 = m1;
    EXPECT_TRUE(AreMatricesEqual(m1, m2));
}

// Тест на корректность присваивания матрицы
TEST(Matrix, CanAssignMatrix) {
    Matrix<double> m1(2), m2(2);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(1, 0) = 3;
    m1(1, 1) = 4;
    m2 = m1;
    EXPECT_TRUE(AreMatricesEqual(m1, m2));
}

// Тест на инициализацию матрицы нулями
TEST(Matrix, InitializesToZero) {
    Matrix<double> m(3);
    for (size_t i = 0; i < m.get_size(); ++i) {
        for (size_t j = 0; j < m.get_size(); ++j) {
            EXPECT_DOUBLE_EQ(0, m(i, j));
        }
    }
}
