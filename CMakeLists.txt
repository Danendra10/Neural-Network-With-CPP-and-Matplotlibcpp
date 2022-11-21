# cmake_minimum_required(VERSION 3.0)

# project(neural_network)

# find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
# find_package(PythonLibs 3.0 REQUIRED)
# include_directories(
#     include
#     ${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

# # add_executable(three_layer src/3L/main.cpp)
# # target_link_libraries(three_layer)

# # add_executable(four_layer src/4L/main.cpp)
# # target_link_libraries(four_layer)

# add_executable(five_layer src/5L/main.cpp)
# target_link_libraries(five_layer ${PYTHON_LIBRARIES})


# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project("example" LANGUAGES CXX)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# find python libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(
    include 
    ${PYTHON3_INCLUDE_DIRS} 
    ${NumPy_INCLUDE_DIRS}
    )

# populate matplotlib repository
include(FetchContent)
FetchContent_Declare(
    matplotlib
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        f23347fca25219d1c42cbb91608b5556814bf572
)
FetchContent_GetProperties(matplotlib)
if(NOT matplotlib_POPULATED)
    FetchContent_Populate(matplotlib)
endif()
include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

# add executable
add_executable(5L src/5L/main.cpp)
add_executable(4L src/4L/main.cpp)
add_executable(3L src/3L/main.cpp)

# link python and numpy
target_link_libraries(5L
    PRIVATE
        ${PYTHON_LIBRARIES}
        Python3::NumPy
)

target_link_libraries(4L
    PRIVATE
        ${PYTHON_LIBRARIES}
        Python3::NumPy
)

target_link_libraries(3L
    PRIVATE
        ${PYTHON_LIBRARIES}
        Python3::NumPy
)