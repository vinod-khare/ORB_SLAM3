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

#include "Viewer.h"

#include "FrameDrawer.h"
#include "MapDrawer.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <exception>

#include <sys/resource.h>
#include <sys/syscall.h>

#define GLFW_INCLUDE_NONE
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
} // namespace

Viewer::Viewer(viewer_control &controls, FrameDrawer &frame_drawer, MapDrawer &map_drawer, float tracked_image_scale, const viewer_settings &settings)
    : _controls(controls), _frame_drawer(frame_drawer), _map_drawer(map_drawer), _settings(settings), _tracked_image_scale(tracked_image_scale)
{
    _dbow_version = controls.dbow_version();
}

void Viewer::Run()
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
            {"MainDockSpace", "Controls", ImGuiDir_Left, 0.2f},
            {"MainDockSpace",   "Camera", ImGuiDir_Down, 0.4f}
        };

        std::array<bool, 7>       layers{true, true, true, true, false, _settings.ground_truth_visible, _settings.camera_trajectory_visible};
        bool                      follow{false};
        bool                      top_view{false};
        bool                      localization{false};
        bool                      step_by_step{false};
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
        params.dockingParams.dockableWindows.emplace_back(
            "Controls", "Controls",
            [&]
            {
                ImGui::SeparatorText("Tracking");
                int dbow_selection = _dbow_version.load() - 2;
                if (ImGui::Combo("DBoW version", &dbow_selection,
                                 "2\0"
                                 "3\0"))
                {
                    _dbow_error   = false;
                    _dbow_version = dbow_selection + 2;
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Reloads the same ORB text vocabulary; tracking waits while loading.");
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
            });
        params.dockingParams.dockableWindows.emplace_back(
            "Map", "MainDockSpace",
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
                        _map_drawer.navigate(dragging ? io.MouseDelta.x : 0, dragging ? io.MouseDelta.y : 0, io.MouseWheel,
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
            auto image = right ? _frame_drawer.DrawRightFrame(_tracked_image_scale) : _frame_drawer.DrawFrame(_tracked_image_scale);
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
        params.dockingParams.dockableWindows.emplace_back(both ? "Left camera" : "Camera", "Camera", [&] { camera_pane(false); });
        if (both)
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
    _stopped  = true;
    _finished = true;
}

void Viewer::process_commands()
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

void Viewer::RequestFinish() { _finish_requested = true; }
void Viewer::RequestStop() { _stop_requested = true; }
bool Viewer::isFinished() const { return _finished; }
bool Viewer::isStopped() const { return _stopped || _finished; }
bool Viewer::exit_requested() const { return _exit_requested; }
void Viewer::Release()
{
    _stop_requested = false;
    _stopped        = _finished.load();
}
} // namespace ORB_SLAM3