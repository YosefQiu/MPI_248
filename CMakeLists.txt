cmake_minimum_required(VERSION 3.10)

project(MyProject)

if(MSVC)
    message(STATUS "Configuring for Windows======")
    include(${CMAKE_CURRENT_LIST_DIR}/CMakeLists_windows.txt)
elseif(UNIX)
    message(STATUS "Configuring for Linux======")
    include(${CMAKE_CURRENT_LIST_DIR}/CMakeLists_linux.txt)
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()