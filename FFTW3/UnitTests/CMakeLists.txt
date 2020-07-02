cmake_minimum_required(VERSION 3.2)

if (CMAKE_BUILD_TYPE MATCHES Test)
    set(PROJECT_NAME X_UT_FFTW3)

    message(${PROJECT_NAME})
    project(${PROJECT_NAME})
    enable_testing()

    file(GLOB_RECURSE SOURCES main.cpp *.cpp *.h *.ipp)
    if (WIN32)
        list(FILTER SOURCES EXCLUDE REGEX "base64.cpp")
    endif()

    add_executable(${PROJECT_NAME} ${SOURCES})
    target_link_libraries(${PROJECT_NAME} FFTW3 SWIFT ${BOOST_STATIC_LIBRARIES})
    if (UNIX)
        target_link_libraries(${PROJECT_NAME} pthread)
    endif()
    message("Linking ${PROJECT_NAME} against FFTW3 SWIFT ${BOOST_STATIC_LIBRARIES}")
endif()