cmake_minimum_required(VERSION 3.0)

project("opti_lib" VERSION 3.0
             DESCRIPTION "A header only c++ library that implements optimization methods to find the minimum of a objective function."
             HOMEPAGE_URL "https://github.com/Jakobimatrix/optilib")


#set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME}
  INTERFACE Eigen3::Eigen)
target_include_directories(${PROJECT_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)


option(TEST_SETTINGS "Build tests for ${PROJECT_NAME}" TRUE)
if(TEST_SETTINGS)
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(executable)
