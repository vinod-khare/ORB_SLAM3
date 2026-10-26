# install dependencies from apt
sudo apt install cmake
sudo apt install ninja-build
sudo apt install libopencv-dev
sudo apt install libeigen3-dev
sudo apt install libcurl4-openssl-dev
sudo apt install libboost-all-dev
sudo apt install libspdlog-dev

# install dependencies from vcpkg
./.vcpkg/vcpkg install 'hello-imgui[glfw-binding,opengl3-binding]' vtk yaml-cpp dbow2 dbow3 g2o ms-gsl spdlog catch2