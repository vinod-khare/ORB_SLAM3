/*
 * File: timestamp.cpp
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * License: see the LICENSE.txt file
 */

#include "DUtils/timestamp.h"

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

using DUtils::timestamp;

timestamp::timestamp(options option) : _seconds(0), _microseconds(0)
{
    if (option & current_time)
    {
        set_to_current_time();
    }
}

bool timestamp::empty() const { return _seconds == 0 && _microseconds == 0; }

void timestamp::set_to_current_time()
{
#ifdef _WIN32
    struct __timeb32 time_buffer;
    _ftime32_s(&time_buffer);
    _seconds      = time_buffer.time;
    _microseconds = time_buffer.millitm * 1000;
#else
    struct timeval now;
    gettimeofday(&now, nullptr);
    _seconds      = now.tv_sec;
    _microseconds = now.tv_usec;
#endif
}

void timestamp::set_time(const std::string &time)
{
    const std::string::size_type point = time.find('.');
    if (point == std::string::npos)
    {
        _seconds      = std::stoul(time);
        _microseconds = 0;
        return;
    }

    const std::string microseconds  = time.substr(point + 1, 6);
    _seconds                        = std::stoul(time.substr(0, point));
    _microseconds                   = std::stoul(microseconds);
    _microseconds                  *= static_cast<unsigned long>(std::pow(10.0, 6.0 - microseconds.length()));
}

void timestamp::set_time(double seconds)
{
    _seconds      = static_cast<unsigned long>(seconds);
    _microseconds = static_cast<unsigned long>((seconds - static_cast<double>(_seconds)) * 1e6);
}

double timestamp::get_float_time() const { return static_cast<double>(_seconds) + static_cast<double>(_microseconds) / 1e6; }

std::string timestamp::get_string_time() const
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6f", get_float_time());
    return buffer;
}

double timestamp::operator-(const timestamp &timestamp) const { return get_float_time() - timestamp.get_float_time(); }

timestamp &timestamp::operator+=(double seconds)
{
    *this = *this + seconds;
    return *this;
}

timestamp &timestamp::operator-=(double seconds)
{
    *this = *this - seconds;
    return *this;
}

timestamp timestamp::operator+(double seconds) const
{
    const auto whole_seconds = static_cast<unsigned long>(std::floor(seconds));
    const auto microseconds  = static_cast<unsigned long>((seconds - whole_seconds) * 1e6);
    return plus(whole_seconds, microseconds);
}

timestamp timestamp::plus(unsigned long seconds, unsigned long microseconds) const
{
    timestamp               result;
    constexpr unsigned long microseconds_per_second = 1000000;
    if (_microseconds + microseconds >= microseconds_per_second)
    {
        result.set_time(_seconds + seconds + 1, _microseconds + microseconds - microseconds_per_second);
    }
    else
    {
        result.set_time(_seconds + seconds, _microseconds + microseconds);
    }
    return result;
}

timestamp timestamp::operator-(double seconds) const
{
    const auto whole_seconds = static_cast<unsigned long>(std::floor(seconds));
    const auto microseconds  = static_cast<unsigned long>((seconds - whole_seconds) * 1e6);
    return minus(whole_seconds, microseconds);
}

timestamp timestamp::minus(unsigned long seconds, unsigned long microseconds) const
{
    timestamp               result;
    constexpr unsigned long microseconds_per_second = 1000000;
    if (_microseconds < microseconds)
    {
        result.set_time(_seconds - seconds - 1, microseconds_per_second - (microseconds - _microseconds));
    }
    else
    {
        result.set_time(_seconds - seconds, _microseconds - microseconds);
    }
    return result;
}

bool timestamp::operator>(const timestamp &timestamp) const { return _seconds > timestamp._seconds || (_seconds == timestamp._seconds && _microseconds > timestamp._microseconds); }

bool timestamp::operator>=(const timestamp &timestamp) const { return _seconds > timestamp._seconds || (_seconds == timestamp._seconds && _microseconds >= timestamp._microseconds); }

bool timestamp::operator<(const timestamp &timestamp) const { return _seconds < timestamp._seconds || (_seconds == timestamp._seconds && _microseconds < timestamp._microseconds); }

bool timestamp::operator<=(const timestamp &timestamp) const { return _seconds < timestamp._seconds || (_seconds == timestamp._seconds && _microseconds <= timestamp._microseconds); }

bool timestamp::operator==(const timestamp &timestamp) const { return _seconds == timestamp._seconds && _microseconds == timestamp._microseconds; }

std::string timestamp::format(bool machine_friendly) const
{
    struct tm    time_parts;
    const time_t time = static_cast<time_t>(get_float_time());
#ifdef _WIN32
    localtime_s(&time_parts, &time);
#else
    localtime_r(&time, &time_parts);
#endif

    char buffer[128];
    strftime(buffer, sizeof(buffer), machine_friendly ? "%Y%m%d_%H%M%S" : "%c", &time_parts);
    return buffer;
}

std::string timestamp::format(double seconds)
{
    const int days           = static_cast<int>(seconds / (24.0 * 3600.0));
    seconds                 -= days * 24.0 * 3600.0;
    const int hours          = static_cast<int>(seconds / 3600.0);
    seconds                 -= hours * 3600.0;
    const int minutes        = static_cast<int>(seconds / 60.0);
    seconds                 -= minutes * 60.0;
    const int whole_seconds  = static_cast<int>(seconds);
    const int microseconds   = static_cast<int>((seconds - whole_seconds) * 1e6);

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