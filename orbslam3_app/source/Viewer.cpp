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
#include "System.h"
#include "Tracking.h"

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

Viewer::Viewer(System *system, FrameDrawer *frame_drawer, MapDrawer *map_drawer, Tracking *tracker, const viewer_settings &settings)
    : _system(system), _frame_drawer(frame_drawer), _map_drawer(map_drawer), _tracker(tracker), _settings(settings), _tracked_image_scale(tracker->GetImageScale())
{
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

        std::array<bool, 5>       layers{true, true, true, true, false};
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
        params.callbacks.ShowStatus = [&] { ImGui::Text("Tracking %.1f FPS  |  UI %.1f FPS", _frame_drawer->get_tracking_fps(), ImGui::GetIO().Framerate); };
        params.dockingParams.dockableWindows.emplace_back("Controls", "Controls",
                                                          [&]
                                                          {
                                                              ImGui::SeparatorText("Tracking");
                                                              if (ImGui::Checkbox("Localization only", &localization))
                                                              {
                                                                  _localization_requested = localization;
                                                              }
                                                              if (ImGui::Checkbox("Step by step", &step_by_step))
                                                              {
                                                                  _tracker->SetStepByStep(step_by_step);
                                                              }
                                                              ImGui::BeginDisabled(!step_by_step);
                                                              if (tool_button(ICON_FA_STEP_FORWARD, "Process next frame"))
                                                              {
                                                                  _tracker->mbStep = true;
                                                              }
                                                              ImGui::EndDisabled();
                                                              ImGui::SameLine();
                                                              if (tool_button(ICON_FA_UNDO, "Reset active map"))
                                                              {
                                                                  localization            = false;
                                                                  _localization_requested = false;
                                                                  _tracker->SetStepByStep(false);
                                                                  step_by_step     = false;
                                                                  _reset_requested = true;
                                                              }
                                                              ImGui::SameLine();
                                                              if (tool_button(ICON_FA_STOP, "Stop playback and save trajectories"))
                                                              {
                                                                  _tracker->SetStepByStep(false);
                                                                  _exit_requested     = true;
                                                                  params.appShallExit = true;
                                                              }
                                                              ImGui::SeparatorText("Map layers");
                                                              ImGui::Checkbox("Map points", &layers[0]);
                                                              ImGui::Checkbox("Keyframes", &layers[1]);
                                                              ImGui::Checkbox("Covisibility graph", &layers[2]);
                                                              ImGui::Checkbox("Inertial graph", &layers[3]);
                                                              ImGui::Checkbox("Local BA colors", &layers[4]);
                                                              ImGui::SeparatorText("View");
                                                              if (ImGui::Checkbox("Follow camera", &follow) && follow)
                                                              {
                                                                  _map_drawer->reset_view(top_view);
                                                              }
                                                              if (tool_button(ICON_FA_CAMERA, "Camera view"))
                                                              {
                                                                  top_view = false;
                                                                  _map_drawer->reset_view(false);
                                                              }
                                                              ImGui::SameLine();
                                                              if (tool_button(ICON_FA_MAP, "Top view"))
                                                              {
                                                                  top_view = true;
                                                                  _map_drawer->reset_view(true);
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
                const auto image   = _map_drawer->render(static_cast<int>(available.x), static_cast<int>(available.y), layers);
                if (follow)
                {
                    _map_drawer->follow_camera(top_view);
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
                        _map_drawer->navigate(dragging ? io.MouseDelta.x : 0, dragging ? io.MouseDelta.y : 0, io.MouseWheel,
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
            auto image = right ? _frame_drawer->DrawRightFrame(_tracked_image_scale) : _frame_drawer->DrawFrame(_tracked_image_scale);
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
            _map_drawer->release();
            glfwMakeContextCurrent(context);
        };
        spdlog::info("Starting ImGui viewer on a low-priority UI thread");
        HelloImGui::Run(params);
    }
    catch (const std::exception &error)
    {
        spdlog::error("Viewer failed: {}", error.what());
        _map_drawer->release();
    }
    _tracker->SetStepByStep(false);
    _stopped  = true;
    _finished = true;
}

void Viewer::process_commands()
{
    const bool localization = _localization_requested;
    if (localization != _localization_active)
    {
        if (localization)
        {
            _system->ActivateLocalizationMode();
        }
        else
        {
            _system->DeactivateLocalizationMode();
        }
        _localization_active = localization;
    }
    if (_reset_requested.exchange(false))
    {
        _system->ResetActiveMap();
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