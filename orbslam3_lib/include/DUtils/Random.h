/*
 * File: Random.h
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010, November 2011
 * Description: manages pseudo-random numbers
 * License: see the LICENSE.txt file
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

namespace DUtils
{

class Random
{
  public:
    class UnrepeatedRandomizer;

    static void                 SeedRand();
    static void                 SeedRandOnce();
    static void                 SeedRand(int seed);
    static void                 SeedRandOnce(int seed);

    template <class T> static T RandomValue() { return static_cast<T>(rand()) / static_cast<T>(RAND_MAX); }

    template <class T> static T RandomValue(T min, T max) { return RandomValue<T>() * (max - min) + min; }

    static int                  RandomInt(int min, int max);

    template <class T> static T RandomGaussianValue(T mean, T sigma)
    {
        T x1;
        T x2;
        T w;
        do
        {
            x1 = static_cast<T>(2) * RandomValue<T>() - static_cast<T>(1);
            x2 = static_cast<T>(2) * RandomValue<T>() - static_cast<T>(1);
            w  = x1 * x1 + x2 * x2;
        } while (w >= static_cast<T>(1) || w == static_cast<T>(0));

        w = std::sqrt((static_cast<T>(-2) * std::log(w)) / w);
        return mean + x1 * w * sigma;
    }

  private:
    static bool m_already_seeded;
};

class Random::UnrepeatedRandomizer
{
  public:
    UnrepeatedRandomizer(int min, int max);
    ~UnrepeatedRandomizer() = default;
    UnrepeatedRandomizer(const UnrepeatedRandomizer &randomizer);
    UnrepeatedRandomizer &operator=(const UnrepeatedRandomizer &randomizer);

    int                   get();
    bool                  empty() const { return m_values.empty(); }
    unsigned int          left() const { return static_cast<unsigned int>(m_values.size()); }
    void                  reset();

  protected:
    void             createValues();

    int              m_min;
    int              m_max;
    std::vector<int> m_values;
};

} // namespace DUtils