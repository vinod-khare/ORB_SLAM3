/*
 * File: timestamp.h
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * License: see the LICENSE.txt file
 */

#pragma once

#include <string>

namespace DUtils
{

class timestamp
{
  public:
    enum options
    {
        none         = 0,
        current_time = 0x1,
        zero         = 0x2
    };

    timestamp(options option = none);
    virtual ~timestamp() = default;

    bool        empty() const;
    void        set_to_current_time();
    void        set_time(unsigned long secs, unsigned long usecs);
    void        get_time(unsigned long &secs, unsigned long &usecs) const;
    void        set_time(const std::string &time);
    void        set_time(double seconds);
    double      get_float_time() const;
    std::string get_string_time() const;

    double     operator-(const timestamp &timestamp) const;
    timestamp &operator+=(double seconds);
    timestamp &operator-=(double seconds);
    timestamp  operator+(double seconds) const;
    timestamp  operator-(double seconds) const;
    bool       operator>(const timestamp &timestamp) const;
    bool       operator>=(const timestamp &timestamp) const;
    bool       operator==(const timestamp &timestamp) const;
    bool       operator<(const timestamp &timestamp) const;
    bool       operator<=(const timestamp &timestamp) const;

    timestamp          plus(unsigned long seconds, unsigned long microseconds) const;
    timestamp          minus(unsigned long seconds, unsigned long microseconds) const;
    std::string        format(bool machine_friendly = false) const;
    static std::string format(double seconds);

  protected:
    unsigned long _seconds;
    unsigned long _microseconds;
};

inline void timestamp::set_time(unsigned long secs, unsigned long usecs)
{
    _seconds      = secs;
    _microseconds = usecs;
}

inline void timestamp::get_time(unsigned long &secs, unsigned long &usecs) const
{
    secs  = _seconds;
    usecs = _microseconds;
}

} // namespace DUtils