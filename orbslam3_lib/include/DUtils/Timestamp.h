/*
 * File: Timestamp.h
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * License: see the LICENSE.txt file
 */

#pragma once

#include <string>

namespace DUtils
{

class Timestamp
{
  public:
    enum tOptions
    {
        NONE         = 0,
        CURRENT_TIME = 0x1,
        ZERO         = 0x2
    };

    Timestamp(tOptions option = NONE);
    virtual ~Timestamp() = default;

    bool        empty() const;
    void        setToCurrentTime();
    void        setTime(unsigned long secs, unsigned long usecs);
    void        getTime(unsigned long &secs, unsigned long &usecs) const;
    void        setTime(const std::string &time);
    void        setTime(double seconds);
    double      getFloatTime() const;
    std::string getStringTime() const;

    double      operator-(const Timestamp &timestamp) const;
    Timestamp  &operator+=(double seconds);
    Timestamp  &operator-=(double seconds);
    Timestamp   operator+(double seconds) const;
    Timestamp   operator-(double seconds) const;
    bool        operator>(const Timestamp &timestamp) const;
    bool        operator>=(const Timestamp &timestamp) const;
    bool        operator==(const Timestamp &timestamp) const;
    bool        operator<(const Timestamp &timestamp) const;
    bool        operator<=(const Timestamp &timestamp) const;

    Timestamp   plus(unsigned long seconds, unsigned long microseconds) const;
    Timestamp   minus(unsigned long seconds, unsigned long microseconds) const;
    std::string Format(bool machine_friendly = false) const;
    static std::string Format(double seconds);

  protected:
    unsigned long m_secs;
    unsigned long m_usecs;
};

inline void Timestamp::setTime(unsigned long secs, unsigned long usecs)
{
    m_secs  = secs;
    m_usecs = usecs;
}

inline void Timestamp::getTime(unsigned long &secs, unsigned long &usecs) const
{
    secs  = m_secs;
    usecs = m_usecs;
}

} // namespace DUtils