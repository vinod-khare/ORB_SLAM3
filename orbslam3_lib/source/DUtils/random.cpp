/*
 * File: random.cpp
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010
 * Description: manages pseudo-random numbers
 * License: see the LICENSE.txt file
 */

#include "DUtils/random.h"

#include "DUtils/timestamp.h"

#include <cstdlib>

bool DUtils::random::_already_seeded = false;

void DUtils::random::seed_rand()
{
    timestamp time;
    time.set_to_current_time();
    srand(static_cast<unsigned int>(time.get_float_time()));
}

void DUtils::random::seed_rand_once()
{
    if (!_already_seeded)
    {
        seed_rand();
        _already_seeded = true;
    }
}

void DUtils::random::seed_rand(int seed) { srand(seed); }

void DUtils::random::seed_rand_once(int seed)
{
    if (!_already_seeded)
    {
        seed_rand(seed);
        _already_seeded = true;
    }
}

int DUtils::random::random_int(int min, int max)
{
    const int range = max - min + 1;
    return static_cast<int>((static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0)) * range) + min;
}

DUtils::random::unrepeated_randomizer::unrepeated_randomizer(int min, int max)
{
    if (min <= max)
    {
        _min = min;
        _max = max;
    }
    else
    {
        _min = max;
        _max = min;
    }
    create_values();
}

DUtils::random::unrepeated_randomizer::unrepeated_randomizer(const unrepeated_randomizer &randomizer) { *this = randomizer; }

int DUtils::random::unrepeated_randomizer::get()
{
    if (empty())
    {
        create_values();
    }

    random::seed_rand_once();
    const int index = random::random_int(0, static_cast<int>(_values.size()) - 1);
    const int value = _values[index];
    _values[index]  = _values.back();
    _values.pop_back();
    return value;
}

void DUtils::random::unrepeated_randomizer::create_values()
{
    const int count = _max - _min + 1;
    _values.resize(count);
    for (int index = 0; index < count; ++index)
    {
        _values[index] = _min + index;
    }
}

void DUtils::random::unrepeated_randomizer::reset()
{
    if (static_cast<int>(_values.size()) != _max - _min + 1)
    {
        create_values();
    }
}

DUtils::random::unrepeated_randomizer &DUtils::random::unrepeated_randomizer::operator=(const unrepeated_randomizer &randomizer)
{
    if (this != &randomizer)
    {
        _min    = randomizer._min;
        _max    = randomizer._max;
        _values = randomizer._values;
    }
    return *this;
}