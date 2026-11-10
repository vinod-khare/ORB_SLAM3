/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "viewer.h"

#include "Tracking.h"
#include "frame_drawer.h"
#include "map_drawer.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <exception>
#include <utility>

#include <sys/resource.h>
#include <sys/syscall.h>

#define GLFW_INCLUDE_NONE
#include <imgui.h>
#include <imgui_stdlib.h>

#include <opencv2/imgproc.hpp>

#include <GLFW/glfw3.h>
#include <hello_imgui/hello_imgui.h>
#include <hello_imgui/icons_font_awesome_4.h>
#include <hello_imgui/texture_gpu.h>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{
namespace
{
void show_image(const HelloImGui::TextureGpuPtr &texture, bool flipped = false)
{
    if (!texture)
    {
        return;
    }
    const auto  available = ImGui::GetContentRegionAvail();
    const float scale     = std::min(available.x / texture->Width, available.y / texture->Height);
    if (scale <= 0)
    {
        return;
    }
    const ImVec2 size(texture->Width * scale, texture->Height * scale);
    const auto   cursor = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(cursor.x + (available.x - size.x) * 0.5f, cursor.y + (available.y - size.y) * 0.5f));
    ImGui::Image(texture->TextureID(), size, ImVec2(0, flipped ? 1 : 0), ImVec2(1, flipped ? 0 : 1));
}

bool tool_button(const char *icon, const char *tooltip)
{
    const bool clicked = ImGui::Button(icon);
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("%s", tooltip);
    }
    return clicked;
}

const char *tracking_state_name(int state)
{
    switch (state)
    {
    case Tracking::SYSTEM_NOT_READY:
        return "System not ready";
    case Tracking::NO_IMAGES_YET:
        return "Waiting for images";
    case Tracking::NOT_INITIALIZED:
        return "Initializing";
    case Tracking::OK:
        return "Tracking";
    case Tracking::RECENTLY_LOST:
        return "Recently lost";
    case Tracking::LOST:
        return "Lost";
    case Tracking::OK_KLT:
        return "KLT tracking";
    default:
        return "Unknown";
    }
}

void show_statistics(const viewer_statistics::snapshot &statistics)
{
    const auto &latest     = statistics.latest;
    const auto &processing = latest.processing;
    const auto &frame      = processing.frame;
    ImGui::Text("Tracking: %s (%d)", tracking_state_name(frame.tracking_state), frame.tracking_state);
    ImGui::Text("SLAM frame ID: %llu | Map: %llu | Change: %d", static_cast<unsigned long long>(frame.frame_id), static_cast<unsigned long long>(frame.map_id), frame.map_change_index);
    ImGui::TextWrapped("Timestamp (Unix UTC): %s", viewer_statistics::format_timestamp(frame.timestamp).c_str());
    ImGui::Text("Source seconds: %.9f", frame.timestamp);
    ImGui::Text("Sequence offset: %s", viewer_statistics::format_duration(statistics.sequence_elapsed_seconds).c_str());
    ImGui::Text("Processing elapsed: %s", viewer_statistics::format_duration(processing.elapsed_seconds).c_str());
    ImGui::Text("Throughput: %.1f FPS (includes waits/pauses)", processing.throughput_fps);

    ImGui::SeparatorText("Camera pose in current map");
    if (latest.camera_to_world)
    {
        const auto position = latest.camera_to_world->translation();
        const auto angles   = viewer_statistics::orientation_degrees(*latest.camera_to_world);
        ImGui::Text("Position XYZ: %.3f, %.3f, %.3f", position.x(), position.y(), position.z());
        ImGui::Text("Roll / pitch / yaw (deg): %.2f, %.2f, %.2f", angles.x(), angles.y(), angles.z());
        ImGui::TextWrapped("Camera-to-world; map units (unscaled monocular is not metric). ZYX Euler convention.");
    }
    else
    {
        ImGui::TextUnformatted("Pose unavailable (not tracking)");
    }

    ImGui::SeparatorText("Current frame");
    ImGui::Text("Detected: %zu | Matched map points: %zu", frame.detected_keypoints, frame.matched_keypoints);
    ImGui::Text("Observations left / right: %zu / %zu", frame.left_keypoints, frame.right_keypoints);
    ImGui::Text("Flow candidates / tracks / inliers: %d / %d / %d", frame.flow_candidates, frame.flow_tracks, frame.flow_inliers);
    ImGui::Text("Flow occupied cells: %d | Flow only: %s | Extracted: %s", frame.flow_occupied_cells, frame.flow_only ? "yes" : "no", frame.features_extracted ? "yes" : "no");
    ImGui::Text("Preprocessing: %.3f ms | Frontend: %.3f ms", frame.preprocessing_ms, frame.frontend_ms);
    ImGui::Text("Extraction: %.3f ms | Stereo: %.3f ms | Flow: %.3f ms", frame.extraction_ms, frame.stereo_matching_ms, frame.optical_flow_ms);
    ImGui::Text("Input wait: %.3f ms | Viewer publication: %.3f ms", latest.input_wait_ms, latest.viewer_publication_ms);

    ImGui::SeparatorText("Per-frame history");
    ImGui::Text("Last %zu processed frames, oldest to newest", statistics.count);
    const auto plot = [&](const char *label, auto value)
    {
        std::array<float, viewer_statistics::history_capacity> values{};
        const auto                                             start = (statistics.next + viewer_statistics::history_capacity - statistics.count) % viewer_statistics::history_capacity;
        for (std::size_t index = 0; index < statistics.count; ++index)
        {
            values[index] = static_cast<float>(value(statistics.history[(start + index) % viewer_statistics::history_capacity]));
        }
        ImGui::TextUnformatted(label);
        ImGui::PushID(label);
        ImGui::PlotLines("##history", values.data(), static_cast<int>(statistics.count), 0, nullptr, 0.f, FLT_MAX, ImVec2(-1, 65));
        ImGui::PopID();
    };
    if (ImGui::BeginTabBar("Statistics plots"))
    {
        if (ImGui::BeginTabItem("Timings (ms)"))
        {
            plot("Frontend", [](const auto &sample) { return sample.frontend_ms; });
            plot("Preprocessing", [](const auto &sample) { return sample.preprocessing_ms; });
            plot("Feature extraction", [](const auto &sample) { return sample.extraction_ms; });
            plot("Stereo matching", [](const auto &sample) { return sample.stereo_matching_ms; });
            plot("Optical flow", [](const auto &sample) { return sample.optical_flow_ms; });
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Feature counts"))
        {
            plot("Detected keypoints", [](const auto &sample) { return sample.detected_keypoints; });
            plot("Matched map points", [](const auto &sample) { return sample.matched_keypoints; });
            plot("Flow tracks", [](const auto &sample) { return sample.flow_tracks; });
            plot("Flow inliers", [](const auto &sample) { return sample.flow_inliers; });
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    if (ImGui::CollapsingHeader("Backend completed work"))
    {
        ImGui::TextWrapped("Independent workers; timings overlap and are not attributable to the current frame.");
        if (ImGui::BeginTable("Backend timings", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("Stage");
            ImGui::TableSetupColumn("Jobs");
            ImGui::TableSetupColumn("Last (ms)");
            ImGui::TableSetupColumn("Total (ms)");
            ImGui::TableHeadersRow();
            const auto row = [](const char *label, const processing_statistics::backend_sample &sample)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(label);
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(sample.jobs));
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", sample.last_ms);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", sample.total_ms);
            };
            row("Local mapping", processing.local_mapping);
            row("Loop closing", processing.loop_closing);
            row("Global BA", processing.global_bundle_adjustment);
            ImGui::EndTable();
        }
    }
}
} // namespace

viewer::viewer(viewer_control &controls, frame_drawer &frame_drawer, map_drawer &map_drawer, float tracked_image_scale, const viewer_settings &settings)
    : _controls(controls), _frame_drawer(frame_drawer), _map_drawer(map_drawer), _settings(settings), _tracked_image_scale(tracked_image_scale)
{
    _dbow_version = controls.dbow_version();
    _paused       = !settings.autostart;
}

void viewer::run()
{
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10) != 0)
    {
        spdlog::warn("Could not lower viewer scheduling priority");
    }
    _stopped = false;
    try
    {
        HelloImGui::RunnerParams params;
        params.appWindowParams.windowTitle              = "ORB-SLAM3";
        params.appWindowParams.windowGeometry.size      = {1440, 900};
        params.appWindowParams.restorePreviousGeometry  = true;
        params.platformBackendType                      = HelloImGui::PlatformBackendType::Glfw;
        params.rendererBackendType                      = HelloImGui::RendererBackendType::OpenGL3;
        params.imGuiWindowParams.defaultImGuiWindowType = HelloImGui::DefaultImGuiWindowType::ProvideFullScreenDockSpace;
        params.imGuiWindowParams.enableViewports        = false;
        params.imGuiWindowParams.showMenuBar            = true;
        params.imGuiWindowParams.showStatusBar          = true;
        params.fpsIdling.enableIdling                   = false;
        params.fpsIdling.fpsMax                         = 30;
        params.iniFolderType                            = HelloImGui::IniFolderType::AppUserConfigFolder;
        params.iniFilename                              = "orbslam3/viewer.ini";
        params.iniFilename_useAppWindowTitle            = false;
        params.dockingParams.dockingSplits              = {
            {"MainDockSpace",   "Controls",  ImGuiDir_Left,  0.2f},
            {"MainDockSpace", "Statistics", ImGuiDir_Right, 0.36f},
            {"MainDockSpace",     "Camera",  ImGuiDir_Down,  0.4f}
        };

        std::array<bool, 7>       layers{true, true, true, true, false, _settings.ground_truth_visible, _settings.camera_trajectory_visible};
        bool                      follow{false};
        bool                      top_view{false};
        bool                      localization{false};
        bool                      step_by_step{false};
        std::string               logging_pattern = _settings.logging_pattern;
        HelloImGui::TextureGpuPtr map_texture;
        HelloImGui::TextureGpuPtr left_texture;
        HelloImGui::TextureGpuPtr right_texture;
        params.callbacks.SetupImGuiStyle = []
        {
            ImGui::StyleColorsLight();
            auto &style          = ImGui::GetStyle();
            style.FrameRounding  = 3;
            style.WindowRounding = 0;
        };
        params.callbacks.ShowGui = [&]
        {
            if (_finish_requested)
            {
                params.appShallExit = true;
            }
            _stopped = _stop_requested.load();
        };
        params.callbacks.ShowStatus = [&] { ImGui::Text("Tracking %.1f FPS  |  UI %.1f FPS", _frame_drawer.get_tracking_fps(), ImGui::GetIO().Framerate); };
        params.dockingParams.dockableWindows.emplace_back("Statistics",
                                                          "Statistics",
                                                          [&]
                                                          {
                                                              if (ImGui::CollapsingHeader("Dataset / sequence", ImGuiTreeNodeFlags_DefaultOpen))
                                                              {
                                                                  ImGui::TextWrapped("%s", _sequence_description.c_str());
                                                              }
                                                              const auto statistics = _statistics.get_snapshot();
                                                              ImGui::Text("Sequence frame: %zu / %zu", statistics.latest.sequence_frame, _sequence_frames);
                                                              if (_sequence_frames > 0)
                                                              {
                                                                  ImGui::ProgressBar(static_cast<float>(statistics.latest.sequence_frame) / static_cast<float>(_sequence_frames),
                                                                                     ImVec2(-1, 0));
                                                              }
                                                              if (statistics.count == 0)
                                                              {
                                                                  ImGui::TextUnformatted("Waiting for the first processed frame");
                                                                  return;
                                                              }
                                                              show_statistics(statistics);
                                                          });
        params.dockingParams.dockableWindows.emplace_back(
            "Controls",
            "Controls",
            [&]
            {
                ImGui::SeparatorText("Tracking");
                const bool paused = _paused.load();
                if (tool_button(paused ? ICON_FA_PLAY : ICON_FA_PAUSE, paused ? "Play" : "Pause"))
                {
                    _controls.set_step_by_step(false);
                    step_by_step = false;
                    _paused      = !paused;
                }
                int dbow_selection = _dbow_version.load() - 2;
                if (ImGui::Combo("DBoW version",
                                 &dbow_selection,
                                 "2\0"
                                 "3\0"))
                {
                    _dbow_error   = false;
                    _dbow_version = dbow_selection + 2;
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Reloads the same binary ORB vocabulary; tracking waits while loading.");
                }
                if (_dbow_error)
                {
                    ImGui::TextWrapped("DBoW switch failed; previous backend retained. See log for details.");
                }
                if (ImGui::Checkbox("Localization only", &localization))
                {
                    _localization_requested = localization;
                }
                if (ImGui::Checkbox("Step by step", &step_by_step))
                {
                    _controls.set_step_by_step(step_by_step);
                    _paused = false;
                }
                ImGui::BeginDisabled(!step_by_step);
                if (tool_button(ICON_FA_STEP_FORWARD, "Process next frame"))
                {
                    _controls.process_next_frame();
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (tool_button(ICON_FA_UNDO, "Reset active map"))
                {
                    localization            = false;
                    _localization_requested = false;
                    _controls.set_step_by_step(false);
                    step_by_step     = false;
                    _reset_requested = true;
                }
                ImGui::SameLine();
                if (tool_button(ICON_FA_STOP, "Stop playback and save trajectories"))
                {
                    _controls.set_step_by_step(false);
                    _exit_requested     = true;
                    params.appShallExit = true;
                }
                ImGui::SeparatorText("Map layers");
                ImGui::Checkbox("Map points", &layers[0]);
                ImGui::Checkbox("Keyframes", &layers[1]);
                ImGui::Checkbox("Covisibility graph", &layers[2]);
                ImGui::Checkbox("Inertial graph", &layers[3]);
                ImGui::Checkbox("Local BA colors", &layers[4]);
                ImGui::BeginDisabled(_settings.ground_truth_path.empty());
                ImGui::Checkbox("Ground truth", &layers[5]);
                ImGui::SameLine();
                ImGui::ColorButton("Ground truth color", ImVec4(0.95f, 0.45f, 0.05f, 1.f), ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_NoTooltip);
                ImGui::EndDisabled();
                ImGui::Checkbox("Camera trajectory", &layers[6]);
                ImGui::SameLine();
                ImGui::ColorButton("Camera trajectory color", ImVec4(0.f, 0.45f, 0.65f, 1.f), ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_NoTooltip);
                ImGui::SeparatorText("View");
                if (ImGui::Checkbox("Follow camera", &follow) && follow)
                {
                    _map_drawer.reset_view(top_view);
                }
                if (tool_button(ICON_FA_CAMERA, "Camera view"))
                {
                    top_view = false;
                    _map_drawer.reset_view(false);
                }
                ImGui::SameLine();
                if (tool_button(ICON_FA_MAP, "Top view"))
                {
                    top_view = true;
                    _map_drawer.reset_view(true);
                }
                ImGui::SameLine();
                if (tool_button(ICON_FA_COLUMNS, "Restore pane layout"))
                {
                    params.dockingParams.layoutReset = true;
                }
                if (ImGui::CollapsingHeader("Logging"))
                {
                    int logging_level = static_cast<int>(spdlog::get_level());
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    if (ImGui::Combo("##logging_level", &logging_level, "trace\0debug\0info\0warning\0error\0critical\0off\0"))
                    {
                        spdlog::set_level(static_cast<spdlog::level::level_enum>(logging_level));
                    }
                    ImGui::TextUnformatted("Pattern");
                    ImGui::SetNextItemWidth(std::max(1.f, ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeightWithSpacing()));
                    const bool apply_pattern = ImGui::InputText("##logging_pattern", &logging_pattern, ImGuiInputTextFlags_EnterReturnsTrue);
                    ImGui::SameLine();
                    if (tool_button(ICON_FA_CHECK, "Apply logging pattern") || apply_pattern)
                    {
                        spdlog::set_pattern(logging_pattern);
                    }
                }
            });
        params.dockingParams.dockableWindows.emplace_back(
            "Map",
            "MainDockSpace",
            [&]
            {
                if (_stopped)
                {
                    return;
                }
                const auto available = ImGui::GetContentRegionAvail();
                if (available.x < 1 || available.y < 1)
                {
                    return;
                }
                auto      *context = glfwGetCurrentContext();
                const auto image   = _map_drawer.render(static_cast<int>(available.x), static_cast<int>(available.y), layers);
                if (follow)
                {
                    _map_drawer.follow_camera(top_view);
                }
                glfwMakeContextCurrent(context);
                cv::Mat rgba;
                cv::cvtColor(image, rgba, cv::COLOR_RGB2RGBA);
                map_texture = HelloImGui::CreateTextureGpuFromRgbaData(rgba.data, rgba.cols, rgba.rows);
                show_image(map_texture, true);
                if (ImGui::IsItemHovered())
                {
                    const auto &io       = ImGui::GetIO();
                    const bool  dragging = ImGui::IsMouseDragging(ImGuiMouseButton_Left) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle) || ImGui::IsMouseDragging(ImGuiMouseButton_Right);
                    if (dragging || io.MouseWheel != 0)
                    {
                        follow = false;
                        _map_drawer.navigate(dragging ? io.MouseDelta.x : 0,
                                             dragging ? io.MouseDelta.y : 0,
                                             io.MouseWheel,
                                             io.KeyShift || ImGui::IsMouseDown(ImGuiMouseButton_Middle) || ImGui::IsMouseDown(ImGuiMouseButton_Right));
                    }
                }
            });
        const auto camera_pane = [&](bool right)
        {
            if (_stopped)
            {
                return;
            }
            auto image = right ? _frame_drawer.draw_right_frame(_tracked_image_scale) : _frame_drawer.draw_frame(_tracked_image_scale);
            if (image.empty())
            {
                return;
            }
            if (_settings.image_view_scale != 1.f)
            {
                cv::resize(image, image, {}, _settings.image_view_scale, _settings.image_view_scale);
            }
            cv::Mat rgba;
            cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
            auto &texture = right ? right_texture : left_texture;
            texture       = HelloImGui::CreateTextureGpuFromRgbaData(rgba.data, rgba.cols, rgba.rows);
            show_image(texture);
        };
        params.dockingParams.dockableWindows.emplace_back(_frame_drawer._both ? "Left camera" : "Camera", "Camera", [&] { camera_pane(false); });
        if (_frame_drawer._both)
        {
            params.dockingParams.dockingSplits.emplace_back("Camera", "RightCamera", ImGuiDir_Right, 0.5f);
            params.dockingParams.dockableWindows.emplace_back("Right camera", "RightCamera", [&] { camera_pane(true); });
        }
        params.callbacks.BeforeExit = [&]
        {
            map_texture.reset();
            left_texture.reset();
            right_texture.reset();
            auto *context = glfwGetCurrentContext();
            _map_drawer.release();
            glfwMakeContextCurrent(context);
        };
        spdlog::info("Starting ImGui viewer on a low-priority UI thread");
        HelloImGui::Run(params);
    }
    catch (const std::exception &error)
    {
        spdlog::error("Viewer failed: {}", error.what());
        _map_drawer.release();
    }
    _controls.set_step_by_step(false);
    _exit_requested = true;
    _stopped        = true;
    _finished       = true;
}

void viewer::process_commands()
{
    const int requested = _dbow_version;
    if (requested != _controls.dbow_version())
    {
        try
        {
            _controls.set_dbow_version(requested);
        }
        catch (const std::exception &error)
        {
            spdlog::error("Could not switch to DBoW{}: {}", requested, error.what());
            _dbow_version = _controls.dbow_version();
            _dbow_error   = true;
        }
    }
    const bool localization = _localization_requested;
    if (localization != _localization_active)
    {
        _controls.set_localization_mode(localization);
        _localization_active = localization;
    }
    if (_reset_requested.exchange(false))
    {
        _controls.reset_active_map();
    }
}

void viewer::set_sequence(std::string description, std::size_t frames)
{
    _sequence_description = std::move(description);
    _sequence_frames      = frames;
}

void viewer::publish_statistics(const viewer_statistics::sample &frame) { _statistics.record(frame); }

void viewer::request_finish() { _finish_requested = true; }
void viewer::request_stop() { _stop_requested = true; }
bool viewer::is_stopped() const { return _stopped || _finished; }
bool viewer::is_exit_requested() const { return _exit_requested; }
bool viewer::is_paused() const { return _paused; }
void viewer::release()
{
    _stop_requested = false;
    _stopped        = _finished.load();
}
} // namespace ORB_SLAM3