#include "yaml_settings.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace
{
void validate_map(const YAML::Node &root, const std::string &name)
{
    const auto node = root[name];

    if (node && !node.IsMap())
    {
        throw std::runtime_error("Settings entry " + name + " must be a map");
    }
}

template <typename T> std::optional<T> read_optional(const YAML::Node &node)
{
    if (!node)
    {
        return std::nullopt;
    }

    return node.as<T>();
}
} // namespace

yaml_settings::yaml_settings(const std::string &settings_path) : _settings_path(settings_path)
{
    std::ifstream stream(settings_path);

    if (!stream)
    {
        throw std::runtime_error("Failed to open settings file: " + settings_path);
    }

    // OpenCV settings use a non-standard %YAML:1.0 directive.
    if (stream.peek() == '%')
    {
        std::string directive;
        std::getline(stream, directive);
    }

    _root = YAML::Load(stream);

    validate_map(_root, "ground_truth");
    validate_map(_root, "camera_trajectory");
    validate_map(_root, "logging");
}

std::optional<bool> yaml_settings::autostart() const { return read_optional<bool>(_root["autostart"]); }

std::optional<int> yaml_settings::dbow_version() const { return read_optional<int>(_root["dbow_version"]); }

std::optional<std::string> yaml_settings::ground_truth_path() const
{
    const auto path_node = _root["ground_truth"]["path"];

    if (!path_node)
    {
        return std::nullopt;
    }

    const auto path = path_node.as<std::string>();

    if (path.empty())
    {
        return std::nullopt;
    }

    return (std::filesystem::absolute(_settings_path).parent_path() / path).lexically_normal().string();
}

std::optional<bool> yaml_settings::ground_truth_visible() const { return read_optional<bool>(_root["ground_truth"]["visible"]); }

std::optional<bool> yaml_settings::camera_trajectory_visible() const { return read_optional<bool>(_root["camera_trajectory"]["visible"]); }

std::optional<spdlog::level::level_enum> yaml_settings::logging_level() const
{
    const auto level_node = _root["logging"]["level"];

    if (!level_node)
    {
        return std::nullopt;
    }

    if (!level_node.IsScalar())
    {
        throw std::runtime_error("Settings entry logging.level must be a string");
    }

    const auto level        = level_node.as<std::string>();
    const auto parsed_level = spdlog::level::from_str(level);

    if (parsed_level == spdlog::level::off && level != "off")
    {
        throw std::runtime_error("logging.level must be trace, debug, info, warning, error, critical or off");
    }

    return parsed_level;
}

std::optional<std::string> yaml_settings::logging_pattern() const
{
    const auto pattern_node = _root["logging"]["pattern"];

    if (!pattern_node)
    {
        return std::nullopt;
    }

    if (!pattern_node.IsScalar())
    {
        throw std::runtime_error("Settings entry logging.pattern must be a string");
    }

    return pattern_node.as<std::string>();
}