cmake_minimum_required(VERSION 3.2)

# Set the project name to the name of the directory replacing any spaces by _
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" ProjectId ${PROJECT_NAME})

message(${PROJECT_NAME})
project(${PROJECT_NAME})

file(GLOB_RECURSE SOURCES *.cpp *.h *.ipp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC SWIFT FFTW3 LEVMAR ${BOOST_STATIC_LIBRARIES})