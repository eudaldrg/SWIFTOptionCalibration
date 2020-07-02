cmake_minimum_required(VERSION 3.15)

# Set the project name to the name of the directory replacing any spaces by _
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" ProjectId ${PROJECT_NAME})

message(${PROJECT_NAME})
project(${PROJECT_NAME})

set(SOURCES
        density_coefficients_calculators.cpp
        density_coefficients_calculators.h
        distributions.cpp
        distributions.h
        known_distribution_contract_combinations.h
        my_math.cpp
        my_math.h
        option_contracts.cpp
        option_contracts.h
        payoff_coefficients_calculators.cpp
        payoff_coefficients_calculators.h
        quick_callibration_swift.cpp
        quick_callibration_swift.h
        sinc_integral_approximators.cpp
        sinc_integral_approximators.h
        swift.cpp
        swift.h
        swift_parameters.cpp
        swift_parameters.h
        )

# For any nested file, replace "/" by  "\"
foreach(source ${SOURCES})
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()

add_library(${PROJECT_NAME} STATIC ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME} PUBLIC ${BOOST_STATIC_LIBRARIES})

add_subdirectory(Z_Exec_CallibrateBS)
add_subdirectory(Z_Exec_CallibrateHeston)
