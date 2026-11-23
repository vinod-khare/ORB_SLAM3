# install dependencies from vcpkg
./.vcpkg install 'hello-imgui[glfw-binding,opengl3-binding]' implot reflectcpp vtk yaml-cpp dbow2 g2o ms-gsl spdlog catch2

# DBoW3 is optional and its vcpkg port is unavailable on macOS.
if [ "$(uname -s)" != "Darwin" ]; then
  ./.vcpkg install dbow3
fi