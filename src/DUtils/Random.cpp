/*
 * File: Random.cpp
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010
 * Description: manages pseudo-random numbers
 * License: see the LICENSE.txt file
 */

#include "DUtils/Random.h"

#include "DUtils/Timestamp.h"

#include <cstdlib>

bool DUtils::Random::m_already_seeded = false;

void DUtils::Random::SeedRand()
{
    Timestamp time;
    time.setToCurrentTime();
    srand(static_cast<unsigned int>(time.getFloatTime()));
}

void DUtils::Random::SeedRandOnce()
{
    if (!m_already_seeded)
    {
        SeedRand();
        m_already_seeded = true;
    }
}

void DUtils::Random::SeedRand(int seed) { srand(seed); }

void DUtils::Random::SeedRandOnce(int seed)
{
    if (!m_already_seeded)
    {
        SeedRand(seed);
        m_already_seeded = true;
    }
}

int DUtils::Random::RandomInt(int min, int max)
{
    const int range = max - min + 1;
    return static_cast<int>((static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0)) * range) + min;
}

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer(int min, int max)
{
    if (min <= max)
    {
        m_min = min;
        m_max = max;
    }
    else
    {
        m_min = max;
        m_max = min;
    }
    createValues();
}

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer(const UnrepeatedRandomizer &randomizer) { *this = randomizer; }

int DUtils::Random::UnrepeatedRandomizer::get()
{
    if (empty())
    {
        createValues();
    }

    Random::SeedRandOnce();
    const int index = Random::RandomInt(0, static_cast<int>(m_values.size()) - 1);
    const int value = m_values[index];
    m_values[index] = m_values.back();
    m_values.pop_back();
    return value;
}

void DUtils::Random::UnrepeatedRandomizer::createValues()
{
    const int count = m_max - m_min + 1;
    m_values.resize(count);
    for (int index = 0; index < count; ++index)
    {
        m_values[index] = m_min + index;
    }
}

void DUtils::Random::UnrepeatedRandomizer::reset()
{
    if (static_cast<int>(m_values.size()) != m_max - m_min + 1)
    {
        createValues();
    }
}

DUtils::Random::UnrepeatedRandomizer &DUtils::Random::UnrepeatedRandomizer::operator=(const UnrepeatedRandomizer &randomizer)
{
    if (this != &randomizer)
    {
        m_min    = randomizer.m_min;
        m_max    = randomizer.m_max;
        m_values = randomizer.m_values;
    }
    return *this;
}