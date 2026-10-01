/*
 * File: Timestamp.cpp
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * License: see the LICENSE.txt file
 */

#include "DUtils/Timestamp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

using DUtils::Timestamp;

Timestamp::Timestamp(tOptions option) : m_secs(0), m_usecs(0)
{
    if (option & CURRENT_TIME)
    {
        setToCurrentTime();
    }
}

bool Timestamp::empty() const { return m_secs == 0 && m_usecs == 0; }

void Timestamp::setToCurrentTime()
{
#ifdef _WIN32
    struct __timeb32 time_buffer;
    _ftime32_s(&time_buffer);
    m_secs  = time_buffer.time;
    m_usecs = time_buffer.millitm * 1000;
#else
    struct timeval now;
    gettimeofday(&now, nullptr);
    m_secs  = now.tv_sec;
    m_usecs = now.tv_usec;
#endif
}

void Timestamp::setTime(const std::string &time)
{
    const std::string::size_type point = time.find('.');
    if (point == std::string::npos)
    {
        m_secs  = std::stoul(time);
        m_usecs = 0;
        return;
    }

    const std::string microseconds  = time.substr(point + 1, 6);
    m_secs                          = std::stoul(time.substr(0, point));
    m_usecs                         = std::stoul(microseconds);
    m_usecs                        *= static_cast<unsigned long>(std::pow(10.0, 6.0 - microseconds.length()));
}

void Timestamp::setTime(double seconds)
{
    m_secs  = static_cast<unsigned long>(seconds);
    m_usecs = static_cast<unsigned long>((seconds - static_cast<double>(m_secs)) * 1e6);
}

double      Timestamp::getFloatTime() const { return static_cast<double>(m_secs) + static_cast<double>(m_usecs) / 1e6; }

std::string Timestamp::getStringTime() const
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6f", getFloatTime());
    return buffer;
}

double     Timestamp::operator-(const Timestamp &timestamp) const { return getFloatTime() - timestamp.getFloatTime(); }

Timestamp &Timestamp::operator+=(double seconds)
{
    *this = *this + seconds;
    return *this;
}

Timestamp &Timestamp::operator-=(double seconds)
{
    *this = *this - seconds;
    return *this;
}

Timestamp Timestamp::operator+(double seconds) const
{
    const auto whole_seconds = static_cast<unsigned long>(std::floor(seconds));
    const auto microseconds  = static_cast<unsigned long>((seconds - whole_seconds) * 1e6);
    return plus(whole_seconds, microseconds);
}

Timestamp Timestamp::plus(unsigned long seconds, unsigned long microseconds) const
{
    Timestamp               result;
    constexpr unsigned long microseconds_per_second = 1000000;
    if (m_usecs + microseconds >= microseconds_per_second)
    {
        result.setTime(m_secs + seconds + 1, m_usecs + microseconds - microseconds_per_second);
    }
    else
    {
        result.setTime(m_secs + seconds, m_usecs + microseconds);
    }
    return result;
}

Timestamp Timestamp::operator-(double seconds) const
{
    const auto whole_seconds = static_cast<unsigned long>(std::floor(seconds));
    const auto microseconds  = static_cast<unsigned long>((seconds - whole_seconds) * 1e6);
    return minus(whole_seconds, microseconds);
}

Timestamp Timestamp::minus(unsigned long seconds, unsigned long microseconds) const
{
    Timestamp               result;
    constexpr unsigned long microseconds_per_second = 1000000;
    if (m_usecs < microseconds)
    {
        result.setTime(m_secs - seconds - 1, microseconds_per_second - (microseconds - m_usecs));
    }
    else
    {
        result.setTime(m_secs - seconds, m_usecs - microseconds);
    }
    return result;
}

bool        Timestamp::operator>(const Timestamp &timestamp) const { return m_secs > timestamp.m_secs || (m_secs == timestamp.m_secs && m_usecs > timestamp.m_usecs); }

bool        Timestamp::operator>=(const Timestamp &timestamp) const { return m_secs > timestamp.m_secs || (m_secs == timestamp.m_secs && m_usecs >= timestamp.m_usecs); }

bool        Timestamp::operator<(const Timestamp &timestamp) const { return m_secs < timestamp.m_secs || (m_secs == timestamp.m_secs && m_usecs < timestamp.m_usecs); }

bool        Timestamp::operator<=(const Timestamp &timestamp) const { return m_secs < timestamp.m_secs || (m_secs == timestamp.m_secs && m_usecs <= timestamp.m_usecs); }

bool        Timestamp::operator==(const Timestamp &timestamp) const { return m_secs == timestamp.m_secs && m_usecs == timestamp.m_usecs; }

std::string Timestamp::Format(bool machine_friendly) const
{
    struct tm    time_parts;
    const time_t time = static_cast<time_t>(getFloatTime());
#ifdef _WIN32
    localtime_s(&time_parts, &time);
#else
    localtime_r(&time, &time_parts);
#endif

    char buffer[128];
    strftime(buffer, sizeof(buffer), machine_friendly ? "%Y%m%d_%H%M%S" : "%c", &time_parts);
    return buffer;
}

std::string Timestamp::Format(double seconds)
{
    const int days                    = static_cast<int>(seconds / (24.0 * 3600.0));
    seconds                          -= days * 24.0 * 3600.0;
    const int hours                   = static_cast<int>(seconds / 3600.0);
    seconds                          -= hours * 3600.0;
    const int minutes                 = static_cast<int>(seconds / 60.0);
    seconds                          -= minutes * 60.0;
    const int          whole_seconds  = static_cast<int>(seconds);
    const int          microseconds   = static_cast<int>((seconds - whole_seconds) * 1e6);

    std::ostringstream stream;
    stream.fill('0');
    bool include_hours = days > 0;
    if (include_hours)
    {
        stream << days << "d ";
    }
    include_hours = include_hours || hours > 0;
    if (include_hours)
    {
        stream << std::setw(2) << hours << ':';
    }
    const bool include_minutes = include_hours || minutes > 0;
    if (include_minutes)
    {
        stream << std::setw(2) << minutes << ':' << std::setw(2);
    }
    stream << whole_seconds;
    if (!include_minutes)
    {
        stream << '.' << std::setw(6) << microseconds;
    }
    return stream.str();
}