
if(NOT "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-subbuild/matplotlib-populate-prefix/src/matplotlib-populate-stamp/matplotlib-populate-gitinfo.txt" IS_NEWER_THAN "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-subbuild/matplotlib-populate-prefix/src/matplotlib-populate-stamp/matplotlib-populate-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-subbuild/matplotlib-populate-prefix/src/matplotlib-populate-stamp/matplotlib-populate-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout "https://github.com/lava/matplotlib-cpp.git" "matplotlib-src"
    WORKING_DIRECTORY "/home/dancoeks/Kuliah/DSEC/NN/build/_deps"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/lava/matplotlib-cpp.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout f23347fca25219d1c42cbb91608b5556814bf572 --
  WORKING_DIRECTORY "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'f23347fca25219d1c42cbb91608b5556814bf572'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-src"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-subbuild/matplotlib-populate-prefix/src/matplotlib-populate-stamp/matplotlib-populate-gitinfo.txt"
    "/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-subbuild/matplotlib-populate-prefix/src/matplotlib-populate-stamp/matplotlib-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/dancoeks/Kuliah/DSEC/NN/build/_deps/matplotlib-subbuild/matplotlib-populate-prefix/src/matplotlib-populate-stamp/matplotlib-populate-gitclone-lastrun.txt'")
endif()

