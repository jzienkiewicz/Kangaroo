cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-std=c++0x -g")
endif()

find_package( Kangaroo 0.1 REQUIRED)
include_directories( ${Kangaroo_INCLUDE_DIRS} )
link_libraries(${Kangaroo_LIBRARIES})

# git clone git://github.com/stevenlovegrove/Pangolin.git
find_package(Pangolin 0.1 QUIET)

# git clone git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus QUIET)

# git clone http://robotics.gwu.edu/git/SceneGraph.git
find_package(SceneGraph 0.1 QUIET)

find_package(CVARS QUIET)

if(Pangolin_FOUND AND Sophus_FOUND AND SceneGraph_FOUND AND CVARS_FOUND)
    include_directories(${Pangolin_INCLUDE_DIRS})
    link_libraries(${Pangolin_LIBRARIES})
    
    include_directories(${Sophus_INCLUDE_DIR})
    link_libraries(${Sophus_LIBRARIES})
    
    include_directories(${SceneGraph_INCLUDE_DIRS})
    link_libraries(${SceneGraph_LIBRARIES})

    add_executable( KinectFusion main.cpp)
endif()
