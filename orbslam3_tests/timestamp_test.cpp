#include "DUtils/timestamp.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("timestamp preserves fractional seconds")
{
    DUtils::timestamp timestamp;
    timestamp.set_time(10UL, 500000UL);

    REQUIRE(timestamp.get_float_time() == 10.5);
}