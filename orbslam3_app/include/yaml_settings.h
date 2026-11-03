#pragma once

#include <optional>
#include <string>

#include <spdlog/common.h>
#include <yaml-cpp/yaml.h>

class yaml_settings
{
  public:
    explicit yaml_settings(const std::string &settings_path);

    std::optional<bool>                      autostart() const;
    std::optional<int>                       dbow_version() const;
    std::optional<std::string>               ground_truth_path() const;
    std::optional<bool>                      ground_truth_visible() const;
    std::optional<bool>                      camera_trajectory_visible() const;
    std::optional<spdlog::level::level_enum> logging_level() const;
    std::optional<std::string>               logging_pattern() const;

  private:
    YAML::Node  _root;
    std::string _settings_path;
};