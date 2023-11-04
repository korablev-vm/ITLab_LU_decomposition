#include "Matrix.h"
#include <gtest.h>

TEST(Matrix, can_create_matrix)
{
	ASSERT_NO_THROW(Matrix m(5));
}