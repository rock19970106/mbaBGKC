cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(mbaBGKC)

set(CMAKE_CXX_STANDARD 11)

#Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

#Eigen
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

#PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#MBA
include_directories("/usr/local/include/mba/include")
link_directories("/usr/local/lib/mba")
link_libraries("/usr/local/lib/mba/libmba.a")

add_executable(mbaBGKC main.cpp)

target_link_libraries(mbaBGKC ${PCL_LIBRARIES})