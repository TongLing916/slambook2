cmake_minimum_required(VERSION 2.8)
project(dense_monocular)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -Wall")
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

############### dependencies ######################
find_package(OpenCV 3.1 REQUIRED)
find_package(Sophus REQUIRED)
find_package(glog REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3")

set(THIRD_PARTY_LIBS
    glog
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES})

set(SRC
    ${PROJECT_SOURCE_DIR}/src/depth_predictor.cc
    ${PROJECT_SOURCE_DIR}/src/util/reader.cc
    ${PROJECT_SOURCE_DIR}/src/util/plotter.cc)
add_library(mapper ${SRC})

add_executable(dense_mapping app/dense_mapping.cpp)
target_link_libraries(dense_mapping
                      mapper
                      ${THIRD_PARTY_LIBS})