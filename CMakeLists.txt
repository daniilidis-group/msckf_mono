cmake_minimum_required(VERSION 2.8.3)
project(msckf_mono)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse4.2 -msse -msse2 -mavx -funroll-loops")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -DEIGEN_NO_DEBUG")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -msse -msse4.2 -msse2")
ENDIF()
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++14 -g")

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

# separate install -- https://github.com/uzh-rpg/fast
find_package(fast REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    image_transport
    cv_bridge
    sensor_msgs
    pcl_conversions
    pcl_ros
    message_generation
      )

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)

# Generate messages in the 'msg' folder
add_message_files(
 FILES
 CamState.msg
 CamStates.msg
 StageTiming.msg
)

# Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
   geometry_msgs  # Or other packages containing msgs
 )

###################################
## catkin specific configuration ##
###################################
catkin_package(
  DEPENDS Eigen
  INCLUDE_DIRS include
  LIBRARIES msckf_mono
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_DIR}
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${fast_INCLUDE_DIRS}
  )

LIST(APPEND LINK_LIBS
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  ${fast_LIBRARIES}
  )

LIST(APPEND MSCKF_SRCS
  src/corner_detector.cpp
  src/ros_interface.cpp
  )

add_library(msckf_mono
  ${MSCKF_SRCS}
  )

target_link_libraries(msckf_mono
  ${LINK_LIBS})

add_executable(asl_msckf datasets/asl_msckf.cpp datasets/asl_readers.cpp)
target_link_libraries(asl_msckf msckf_mono ${LINK_LIBS})

add_executable(asl_msckf_no_ground_truth datasets/asl_msckf_no_ground_truth.cpp datasets/asl_readers.cpp)
target_link_libraries(asl_msckf_no_ground_truth msckf_mono ${LINK_LIBS})

add_executable(msckf_mono_node nodes/msckf_mono_node.cpp)
target_link_libraries(msckf_mono_node msckf_mono ${LINK_LIBS})

# add_executable(pcv_msckf datasets/penncosy_msckf.cpp datasets/penncosy_readers.cpp)
# target_link_libraries(pcv_msckf ${LINK_LIBS} msckf_mono)
