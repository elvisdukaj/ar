cmake_minimum_required(VERSION 3.11)

project(markergen)

set(HEADERS_FILES "include/options.hpp")
set(SOURCE_FILES "src/options.cpp" "src/main.cpp")
set(SOURCES ${HEADERS_FILES} ${SOURCE_FILES} ${RESOURCES_FILES})

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(OpenCV REQUIRED highgui)
find_package(Boost COMPONENTS program_options REQUIRED)

target_include_directories(${PROJECT_NAME}  PRIVATE include)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME}
PUBLIC
    Boost::dynamic_linking
    Boost::program_options
    ${OpenCV_LIBS}
    )
