cmake_minimum_required(VERSION 3.9)
project(my_lime)

set(CMAKE_CXX_STANDARD 14)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )


#查找需要依赖的库环境
find_package( OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package( FFTW3 REQUIRED)

#启用多核心编程
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


# 设置编译选项以针对 ARM 架构进行优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
endif()

#添加编译器选项
set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS -mfpu=neon)

#库路径
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        /usr/local/include/eigen3/
        ${FFTW3_INCLUDE_DIRS}

)


file(GLOB SOURCES "src/*.*" "main.cpp")
add_executable(my_lime ${SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${FFTW3_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

message(STATUS "${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "OpenCV library status:")
message(STATUS "FFTW3_DIR:${FFTW3_INCLUDE_DIRS}")
message(STATUS "config: ${OpenCV_DIR}")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")


