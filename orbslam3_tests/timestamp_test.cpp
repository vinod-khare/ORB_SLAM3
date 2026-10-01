#include "DUtils/Timestamp.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Timestamp preserves fractional seconds")
{
    DUtils::Timestamp timestamp;
    timestamp.setTime(10UL, 500000UL);

    REQUIRE(timestamp.getFloatTime() == 10.5);
}