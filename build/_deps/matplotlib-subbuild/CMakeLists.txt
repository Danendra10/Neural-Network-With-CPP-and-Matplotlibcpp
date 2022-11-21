# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(matplotlib-populate NONE)

include(ExternalProject)
ExternalProject_Add(matplotlib-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/lava/matplotlib-cpp.git" "GIT_TAG" "f23347fca25219d1c42cbb91608b5556814bf572"
                    SOURCE_DIR          "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-src"
                    BINARY_DIR          "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
