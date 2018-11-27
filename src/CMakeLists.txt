CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )


find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# g2o
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS})
FIND_PACKAGE(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV 2.4.9 REQUIRED)
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

include_directories( "/usr/include/eigen3" )

find_package(Sophus REQUIRED)
INCLUDE_DIRECTORIES(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# pcl
find_package( PCL REQUIRED COMPONENT common io visualization filters)
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

include_directories(Eigen)

add_library(slambase slamBase.cpp)
target_link_libraries(slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(generate_pointcloud generatePointCloud.cpp)
target_link_libraries(generate_pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(detectFeatures detectFeatures.cpp)
target_link_libraries(detectFeatures slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(jointPointCloud jointPointCloud.cpp)
target_link_libraries(jointPointCloud slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(visualOdometry visualOdometry.cpp)
target_link_libraries(visualOdometry slambase ${OpenCV_LIBS} ${PCL_LIBRARIES}
                      ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES})

add_executable(slamEnd slamEnd.cpp)
target_link_libraries(slamEnd slambase ${OpenCV_LIBS} ${PCL_LIBRARIES}
                      ${Sophus_LIBRARIES} g2o_core g2o_types_slam3d g2o_solver_csparse
                      g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY} ${Pangolin_LIBRARIES})

add_executable(slam slam.cpp)
target_link_libraries(slam slambase ${OpenCV_LIBS} ${PCL_LIBRARIES}
        ${Sophus_LIBRARIES} g2o_core g2o_types_slam3d g2o_solver_csparse
        g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY} ${Pangolin_LIBRARIES} glog)


#FIND_PACKAGE(OpenCV REQUIRED)
#INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
#
add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(Dense_Mono Dense_Mono.cpp)
target_link_libraries(Dense_Mono slambase ${OpenCV_LIBS} ${PCL_LIBRARIES}
        ${Sophus_LIBRARIES} g2o_core g2o_types_slam3d g2o_solver_csparse
        g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY} ${Pangolin_LIBRARIES})