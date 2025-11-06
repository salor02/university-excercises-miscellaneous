cmake_minimum_required(VERSION 3.10)
project(AlignTrajectories)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(align align_trajectory.cpp)
target_link_libraries(align Eigen3::Eigen Ceres::ceres)
set(CMAKE_CXX_STANDARD 17)
