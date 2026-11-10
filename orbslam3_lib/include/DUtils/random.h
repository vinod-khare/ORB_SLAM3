/*
 * File: random.h
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

class random
{
  public:
    class unrepeated_randomizer;

    static void seed_rand();
    static void seed_rand_once();
    static void seed_rand(int seed);
    static void seed_rand_once(int seed);

    template <class T> static T random_value() { return static_cast<T>(rand()) / static_cast<T>(RAND_MAX); }

    template <class T> static T random_value(T min, T max) { return random_value<T>() * (max - min) + min; }

    static int random_int(int min, int max);

    template <class T> static T random_gaussian_value(T mean, T sigma)
    {
        T x1;
        T x2;
        T w;
        do
        {
            x1 = static_cast<T>(2) * random_value<T>() - static_cast<T>(1);
            x2 = static_cast<T>(2) * random_value<T>() - static_cast<T>(1);
            w  = x1 * x1 + x2 * x2;
        } while (w >= static_cast<T>(1) || w == static_cast<T>(0));

        w = std::sqrt((static_cast<T>(-2) * std::log(w)) / w);
        return mean + x1 * w * sigma;
    }

  private:
    static bool _already_seeded;
};

class random::unrepeated_randomizer
{
  public:
    unrepeated_randomizer(int min, int max);
    ~unrepeated_randomizer() = default;
    unrepeated_randomizer(const unrepeated_randomizer &randomizer);
    unrepeated_randomizer &operator=(const unrepeated_randomizer &randomizer);

    int          get();
    bool         empty() const { return _values.empty(); }
    unsigned int left() const { return static_cast<unsigned int>(_values.size()); }
    void         reset();

  protected:
    void create_values();

    int              _min;
    int              _max;
    std::vector<int> _values;
};

} // namespace DUtils