cmake_minimum_required(VERSION 3.11)

project(calibration_filter LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

add_executable(${PROJECT_NAME}
    "include/abstractopencvrunnablefilter.h"
    "include/calibrationfilter.h"
    "src/abstractopencvrunnablefilter.cpp"
    "src/calibrationfilter.cpp"
    "src/main.cpp"
    "resource/qml.qrc"
)

find_package(Qt5 COMPONENTS Core Quick Multimedia REQUIRED)
find_package(OpenCV REQUIRED imgproc calib3d)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME}
PRIVATE
    Qt5::Core
    Qt5::Quick
    Qt5::Multimedia
    ${OpenCV_LIBS}
    )
