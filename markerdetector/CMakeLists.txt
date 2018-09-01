cmake_minimum_required(VERSION 3.5)

project(markerdetector LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Core Quick Multimedia REQUIRED)
find_package(OpenCV REQUIRED imgproc calib3d)
find_package(Boost REQUIRED)

add_executable(${PROJECT_NAME}
    "include/abstractopencvrunnablefilter.h"
    "include/marker.h"
    "include/markerdetector.h"
    "include/markerdetectorfilter.h"
    "src/abstractopencvrunnablefilter.cpp"
    "src/marker.cpp"
    "src/markerdetector.cpp"
    "src/markerdetectorfilter.cpp"
    "src/main.cpp"
    "resource/qml.qrc"
    )

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME}
    Boost::boost
    Qt5::Core
    Qt5::Quick
    Qt5::Multimedia
    ${OpenCV_LIBS}
)
