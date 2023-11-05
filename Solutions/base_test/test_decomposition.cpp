#include "Matrix.h"
#include <gtest.h>

// ��������������� ������� ��� �������� ��������� ���� ������
bool AreMatricesEqual(const Matrix& m1, const Matrix& m2) {
    if (m1.get_size() != m2.get_size()) return false;
    for (size_t i = 0; i < m1.get_size(); ++i) {
        for (size_t j = 0; j < m1.get_size(); ++j) {
            if (std::abs(m1[i][j] - m2[i][j]) > 1e-6) { // �������� � ��������� ������������
                return false;
            }
        }
    }
    return true;
}

// ���� �� ������������ LU-����������
TEST(Matrix, LUDecompositionCorrectness) {
    Matrix m(3);
    m[0][0] = 2; m[0][1] = -1; m[0][2] = -2;
    m[1][0] = -4; m[1][1] = 6; m[1][2] = 3;
    m[2][0] = -4; m[2][1] = -2; m[2][2] = 8;

    Matrix L(3), U(3);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}

// ���� �� LU-���������� � ������� ��������� �� ������� ���������
TEST(Matrix, LUDecompositionZeroDiagonalElement) {
    Matrix m(2);
    m[0][0] = 0; m[0][1] = -1;
    m[1][0] = -4; m[1][1] = 6;

    Matrix L(2), U(2);
    EXPECT_THROW(m.LU_Decomposition(L, U), std::runtime_error);
}

// ���� �� LU-���������� � ��������� ��������
TEST(Matrix, LUDecompositionIdentityMatrix) {
    Matrix m(3);
    m[0][0] = 1; m[0][1] = 0; m[0][2] = 0;
    m[1][0] = 0; m[1][1] = 1; m[1][2] = 0;
    m[2][0] = 0; m[2][1] = 0; m[2][2] = 1;

    Matrix L(3), U(3);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    // L � U ������ ���� ���������� ���������
    EXPECT_TRUE(AreMatricesEqual(m, L));
    EXPECT_TRUE(AreMatricesEqual(m, U));
}

// ���� �� LU-���������� � ��������, ���������� ������������� ��������
TEST(Matrix, LUDecompositionWithNegativeElements) {
    Matrix m(2);
    m[0][0] = -1; m[0][1] = -2;
    m[1][0] = -3; m[1][1] = -4;

    Matrix L(2), U(2);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}

// ���� �� LU-���������� � ��������, ���������� ������� �����
TEST(Matrix, LUDecompositionWithFractionalElements) {
    Matrix m(2);
    m[0][0] = 0.5; m[0][1] = 1.5;
    m[1][0] = 1.2; m[1][1] = 2.3;

    Matrix L(2), U(2);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}

// ���� �� LU-���������� � �������� �������� �������
TEST(Matrix, LUDecompositionLargeMatrix) {
    const size_t size = 10; // ���������� ������� �������� 10x10
    Matrix m(size);
    // ���������� ������� ���������� ����������
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            m[i][j] = rand() % 100;
        }
    }

    Matrix L(size), U(size);
    ASSERT_NO_THROW(m.LU_Decomposition(L, U));

    Matrix product = L.multiply(U);
    EXPECT_TRUE(AreMatricesEqual(m, product));
}


TEST(Matrix, CanCreateMatrix) 
{
    ASSERT_NO_THROW(Matrix m(5));
}
// ���� �� ������ ��������� �������
TEST(Matrix, CorrectSize) {
    Matrix m(5);
    EXPECT_EQ(5, m.get_size());
}

// ���� �� ������������� ������� ������
TEST(Matrix, InitializesToZero) {
    Matrix m(3);
    for (size_t i = 0; i < m.get_size(); ++i) {
        for (size_t j = 0; j < m.get_size(); ++j) {
            EXPECT_DOUBLE_EQ(0, m[i][j]);
        }
    }
}
// ���� �� ����������� ��������� �������� �������
TEST(Matrix, CanSetValues) {
    Matrix m(3);
    m[1][1] = 5.5;
    EXPECT_DOUBLE_EQ(5.5, m[1][1]);
}
// ���� �� ������������ ��������� ������
TEST(Matrix, CanMultiplyMatrices) {
    Matrix m1(2), m2(2);
    m1[0][0] = 1; m1[0][1] = 2;
    m1[1][0] = 3; m1[1][1] = 4;

    m2[0][0] = 5; m2[0][1] = 6;
    m2[1][0] = 7; m2[1][1] = 8;

    Matrix result = m1.multiply(m2);
    EXPECT_DOUBLE_EQ(result[0][0], 19);
    EXPECT_DOUBLE_EQ(result[0][1], 22);
    EXPECT_DOUBLE_EQ(result[1][0], 43);
    EXPECT_DOUBLE_EQ(result[1][1], 50);
}
// ���� �� ������ ���������� ��� ��������� ������ ������� �������
TEST(Matrix, ThrowsOnMultiplyDifferentSizes) {
    Matrix m1(2), m2(3);
    EXPECT_THROW(m1.multiply(m2), std::invalid_argument);
}
// ���� �� ������ ���������� ��� ������������ �������� ��� LU-����������
TEST(Matrix, ThrowsOnLUDecompositionWithIncorrectSizes) {
    Matrix m(3), L(2), U(2);
    EXPECT_THROW(m.LU_Decomposition(L, U), std::invalid_argument);
}
TEST(Matrix, CanOutputMatrix) {
    Matrix m(2);
    m[0][0] = 1; m[0][1] = 2;
    m[1][0] = 3; m[1][1] = 4;
    std::stringstream ss;
    ss << m;
    std::string expected = "1 2 \n3 4 \n";
    EXPECT_EQ(ss.str(), expected);
}
// ���� �� ������������ ������� � ��������� �������
TEST(Matrix, CanAccessElements) {
    Matrix m(2);
    m[0][0] = 1;
    const Matrix& cm = m;
    EXPECT_DOUBLE_EQ(1, cm[0][0]);
}

// ���� �� ������������ ����������� �������
TEST(Matrix, CanCopyMatrix) {
    Matrix m1(2);
    m1[0][0] = 1; m1[0][1] = 2;
    m1[1][0] = 3; m1[1][1] = 4;
    Matrix m2 = m1;
    EXPECT_EQ(m1[0][0], m2[0][0]);
    EXPECT_EQ(m1[0][1], m2[0][1]);
    EXPECT_EQ(m1[1][0], m2[1][0]);
    EXPECT_EQ(m1[1][1], m2[1][1]);
}

// ���� �� ������������ ������������ �������
TEST(Matrix, CanAssignMatrix) {
    Matrix m1(2), m2(2);
    m1[0][0] = 1; m1[0][1] = 2;
    m1[1][0] = 3; m1[1][1] = 4;
    m2 = m1;
    EXPECT_EQ(m1[0][0], m2[0][0]);
    EXPECT_EQ(m1[0][1], m2[0][1]);
    EXPECT_EQ(m1[1][0], m2[1][0]);
    EXPECT_EQ(m1[1][1], m2[1][1]);
}