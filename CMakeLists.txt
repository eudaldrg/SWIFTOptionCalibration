cmake_minimum_required(VERSION 3.8.2)

#execute_process(
#  COMMAND tail -n 1 ReleaseNotes.txt
#  COMMAND sed -e "s|v\\(.*\\) - .*|\\1|"
#  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#  OUTPUT_VARIABLE RELEASE_VERSION
#)
#
#execute_process(
#  COMMAND tail -n 1 DevNotes.txt
#  COMMAND sed -e "s|v\\(.*\\) - .*|\\1|"
#  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#  OUTPUT_VARIABLE MASTER_VERSION
#)

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set to 1 to use the address sanitizer.
set(ADDRESS_SANITIZER 0)

set(CMAKE_AUTORCC ON)
set(SOUND_RESOURCES "${CMAKE_SOURCE_DIR}/VisuNG/Resources/sounds.qrc")
set(IMAGE_RESOURCES "${CMAKE_SOURCE_DIR}/VisuNG/Resources/images.qrc")

# Find version or branch.
if (DEFINED "GIT_BRANCH")
    set(VERSION ${GIT_BRANCH})
    if ($ENV{CLION_IDE})
        message("Running in CLion")
        if (DEFINED "CLION_USING_CLANG")
            set(CMAKE_CXX_COMPILER "/usr/bin/clang++-7")
        else()
            set(CMAKE_CXX_COMPILER_ID "gcc")
        endif()
    else()
        message("Running in Terminal for $ENV{USER}")
        if (WIN32)
            set(CMAKE_CXX_COMPILER_ID "MSVC")
        else()
            if(EXISTS "${CMAKE_SOURCE_DIR}/compile_with_clang" OR ADDRESS_SANITIZER)
                set(CMAKE_C_COMPILER "/usr/bin/clang-7")
                set(CMAKE_CXX_COMPILER "/usr/bin/clang++-7")
                set(CMAKE_CXX_COMPILER_ID "Clang")
                set(CMAKE_CXX_COMPILER_VERSION "7.0.0")
            else()
                set(CMAKE_CXX_COMPILER_ID "gcc")
            endif()
        endif()
    endif()
else()
  set(GIT_BRANCH "NOBRANCH")
  set(VERSION "NOVERSION")
endif()

# Detect operative system.
if (WIN32)
	set(OS "Windows")
elseif (UNIX)
	set(OS "Unix")
else()
	set(OS "Unknown")
endif()

# At the user's choice, export a compilation database to be used for code completion features.
if ($ENV{VIM_SUPPORT})
    message("Export compilation database.")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)
endif()

if (WIN32)
    set(PCH_ENABLE 0 CACHE STRING "" FORCE)
else()
    set(PCH_ENABLE 1 CACHE STRING "" FORCE)
endif()

#message(${MASTER_VERSION})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules/")
#include(GetGitRevisionDescription)
#include(PrecompiledHeaders)

if (NOT (EXISTS ${CMAKE_SOURCE_DIR}/git_sha_files))
    file(MAKE_DIRECTORY "git_sha_files")
endif()

#set(CMAKE_VERBOSE_MAKEFILE on)

#### Build configuration ####
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("Setting up mode ${CMAKE_BUILD_TYPE} for branch ${GIT_BRANCH} with version ${VERSION} and compiler ${CMAKE_CXX_COMPILER_ID} with C++${CMAKE_CXX_STANDARD}")

if(ADDRESS_SANITIZER)
    # We never link TC MALLOC with the address sanitizer
    set(LINK_TC_MALLOC 0)
else()
    # Set to 0 to avoid linking tcmalloc
    set(LINK_TC_MALLOC 1)
endif()

if (WIN32)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;GRelease;Test" CACHE STRING "" FORCE)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /MP /EHsc -DVISUALIZATION -DREVEALING_STRINGS -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -D_WINSOCK_DEPRECATED_NO_WARNINGS -DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /Ob0 /MDd /RTC1 /bigobj -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/Gy /GF /Oy /Oi /MD /bigobj -DNDEBUG")
    set(CMAKE_CXX_FLAGS_GRELEASE "/Zi /bigobj /Gy /GF /Oy /Oi -DNDEBUG /W4 /WX /MP /EHsc -DVISUALIZATION -DREVEALING_STRINGS -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -D_WINSOCK_DEPRECATED_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS_TEST "/Zi /Od /Ob0 /MDd /RTC1 /bigobj -DDEBUG -DEXTENDED_DEBUG -DCME_MDP_AUTOCERT -DCME_FIX_AUTOCERT -DTEST_MODE")

    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO /SAFESEH:NO")
    set(CMAKE_EXE_LINKER_FLAGS_GRELEASE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")
    set(CMAKE_SHARED_LINKER_FLAGS_GRELEASE "${CMAKE_EXE_LINKER_FLAGS}")

	set(LINK_TC_MALLOC 0)
else()
    set(CMAKE_CONFIGURATION_TYPES "Debug;ExtendedDebug;Release;Profile;GRelease;RRelease;RGLogRelease;RDebug;GProduction;DebugProduction;Test" CACHE STRING "" FORCE)
    set(CMAKE_CXX_TCMALLOC_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    if (${CMAKE_BUILD_TYPE} MATCHES "GRelease" OR ${CMAKE_BUILD_TYPE} MATCHES "Test" OR ${CMAKE_BUILD_TYPE} MATCHES "Debug")
        set(LINK_TC_MALLOC 0)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wshadow -Wuninitialized -Winit-self -Wno-unknown-pragmas -Wno-deprecated-declarations -ftemplate-backtrace-limit=0")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fpic -g -DDEBUG -DVISUALIZATION -DREVEALING_STRINGS")
    set(CMAKE_CXX_FLAGS_EXTENDEDDEBUG "-O0 -fpic -g -DDEBUG -DEXTENDED_DEBUG -DVISUALIZATION -DREVEALING_STRINGS")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fpic -fomit-frame-pointer -DNDEBUG -DQT_NO_DEBUG -DFILELOG_MAX_LEVEL=LogL::Info -DVISUALIZATION -DREVEALING_STRINGS")
    set(CMAKE_CXX_FLAGS_PROFILE "-O3 -fpic -g -DNDEBUG -DQT_NO_DEBUG -DVISUALIZATION -DREVEALING_STRINGS -DPROFILE")
    set(CMAKE_CXX_FLAGS_GRELEASE "-O1 -fpic -g -DNDEBUG -DQT_NO_DEBUG -DVISUALIZATION -DREVEALING_STRINGS") #-Wno-maybe-uninitialized
    set(CMAKE_CXX_FLAGS_RRELEASE "-O3 -fpic -DRSIMULATOR -DNDEBUG -DQT_NO_DEBUG -DFILELOG_MAX_LEVEL=LogL::Info -DVISUALIZATION -DREVEALING_STRINGS -DUSING_RCPP")
    set(CMAKE_CXX_FLAGS_RGLOGRELEASE "-O1 -g -fpic -DRSIMULATOR -DNDEBUG -DQT_NO_DEBUG -DVISUALIZATION -DREVEALING_STRINGS -DUSING_RCPP")
    set(CMAKE_CXX_FLAGS_RDEBUG "-O0 -fpic -g -DDEBUG -DRSIMULATOR -DVISUALIZATION -DREVEALING_STRINGS -DUSING_RCPP")
    set(CMAKE_CXX_FLAGS_GPRODUCTION "-march=ivybridge -mmmx -mno-3dnow -msse -msse2 -msse3 -mssse3 -mno-sse4a -mcx16 -msahf -mno-movbe -maes -mno-sha -mpclmul -mpopcnt -mno-abm -mno-lwp -mno-fma -mno-fma4 -mno-xop -mno-bmi -mno-sgx -mno-bmi2 -mno-pconfig -mno-wbnoinvd -mno-tbm -mavx -mno-avx2 -msse4.2 -msse4.1 -mno-lzcnt -mno-rtm -mno-hle -mrdrnd -mf16c -mfsgsbase -mno-rdseed -mno-prfchw -mno-adx -mfxsr -mxsave -mxsaveopt -mno-avx512f -mno-avx512er -mno-avx512cd -mno-avx512pf -mno-prefetchwt1 -mno-clflushopt -mno-xsavec -mno-xsaves -mno-avx512dq -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi -mno-avx5124fmaps -mno-avx5124vnniw -mno-clwb -mno-mwaitx -mno-clzero -mno-pku -mno-rdpid -mno-gfni -mno-shstk -mno-avx512vbmi2 -mno-avx512vnni -mno-vaes -mno-vpclmulqdq -mno-avx512bitalg -mno-movdiri -mno-movdir64b --param l1-cache-size=32 --param l1-cache-line-size=64 --param l2-cache-size=8192 -mtune=ivybridge -fstack-protector-strong -Wformat -Wformat-security -O3 -g -fpic -fno-omit-frame-pointer -DNDEBUG -DQT_NO_DEBUG -DVISUALIZATION -DFILELOG_MAX_LEVEL=LogL::Info") # Use "gcc -march=native -E -v - </dev/null 2>&1 | grep cc1" on the target machine to get compile flags
    set(CMAKE_CXX_FLAGS_DEBUGPRODUCTION "-O0 -fpic -g -DDEBUG -DVISUALIZATION")
    # As in Test we make A_Utils depend on Qt at link time, and Qt adds -fPIC to compilation flags of everything that depends on it, adding -fPIC here only means a change to those libraries that don't have a link dependency on A_Utils. Needed for the precompiled headers.
    set(CMAKE_CXX_FLAGS_TEST "-O0 -fPIC -fno-access-control -g -DDEBUG -DEXTENDED_DEBUG -DVISUALIZATION -DREVEALING_STRINGS -DCME_MDP_AUTOCERT -DCME_FIX_AUTOCERT -DTEST_MODE")
endif()

if ($ENV{CLION_IDE})
    message("Ignoring Clang flags for terminal")
    if (DEFINED "CLION_USING_CLANG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=750")
    endif()
else()
    if (DEFINED "CMAKE_CXX_COMPILER_ID" AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    #    message(${CMAKE_CXX_COMPILER_ID})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=750")
        if (ADDRESS_SANITIZER)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
        endif()

        set(CMAKE_CXX_FLAGS_GPRODUCTION "-O3 -g -fpic -fno-omit-frame-pointer -DNDEBUG -DQT_NO_DEBUG -DFILELOG_MAX_LEVEL=LogL::Info")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -ftemplate-depth=512")
    endif()
endif()

project(Xanadu)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#### Build directory ####
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build_x64/Debug/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_EXTENDEDDEBUG "${CMAKE_SOURCE_DIR}/build_x64/ExtendedDebug/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build_x64/Release/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_PROFILE "${CMAKE_SOURCE_DIR}/build_x64/Profile/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_GRELEASE "${CMAKE_SOURCE_DIR}/build_x64/GRelease/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RRELEASE "${CMAKE_SOURCE_DIR}/build_x64/RRelease/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RGLOGRELEASE "${CMAKE_SOURCE_DIR}/build_x64/RGLogRelease/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RDEBUG "${CMAKE_SOURCE_DIR}/build_x64/RDebug/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_GPRODUCTION "${CMAKE_SOURCE_DIR}/build_x64/GProduction/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUGPRODUCTION "${CMAKE_SOURCE_DIR}/build_x64/DebugProduction/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_TEST "${CMAKE_SOURCE_DIR}/build_x64/Test/bin")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build_x64/Debug/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_EXTENDEDDEBUG "${CMAKE_SOURCE_DIR}/build_x64/ExtendedDebug/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build_x64/Release/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_PROFILE "${CMAKE_SOURCE_DIR}/build_x64/Profile/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_GRELEASE "${CMAKE_SOURCE_DIR}/build_x64/GRelease/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RRELEASE "${CMAKE_SOURCE_DIR}/build_x64/RRelease/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RGLOGRELEASE "${CMAKE_SOURCE_DIR}/build_x64/RGLogRelease/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RDEBUG "${CMAKE_SOURCE_DIR}/build_x64/RDebug/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_GPRODUCTION "${CMAKE_SOURCE_DIR}/build_x64/GProduction/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUGPRODUCTION "${CMAKE_SOURCE_DIR}/build_x64/DebugProduction/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_TEST "${CMAKE_SOURCE_DIR}/build_x64/Test/lib")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build_x64/Debug/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_EXTENDEDDEBUG "${CMAKE_SOURCE_DIR}/build_x64/ExtendedDebug/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build_x64/Release/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_PROFILE "${CMAKE_SOURCE_DIR}/build_x64/Profile/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_GRELEASE "${CMAKE_SOURCE_DIR}/build_x64/GRelease/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RRELEASE "${CMAKE_SOURCE_DIR}/build_x64/RRelease/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RGLOGRELEASE "${CMAKE_SOURCE_DIR}/build_x64/RGLogRelease/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RDEBUG "${CMAKE_SOURCE_DIR}/build_x64/RDebug/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_GPRODUCTION "${CMAKE_SOURCE_DIR}/build_x64/GProduction/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUGPRODUCTION "${CMAKE_SOURCE_DIR}/build_x64/DebugProduction/slib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_TEST "${CMAKE_SOURCE_DIR}/build_x64/Test/slib")

#### Output compilation times
if(EXISTS "${CMAKE_SOURCE_DIR}/compile_times")
    message("Compilation times printed to standard error")
    set(CMAKE_CXX_COMPILE_OBJECT "exec time -f 'elapsed=%e\tuser=%U\tsystem=%S\t<SOURCE>\\n%C' ${CMAKE_CXX_COMPILE_OBJECT}") # %C prints the full compilation command with flags
endif()

#### Boost and Qt directories ####
if (WIN32)
    set(BOOST_ROOT "C:/Sys/boost_1_68_0")
    set(CMAKE_PREFIX_PATH "C:\\Qt\\5.9.1\\msvc2017_64\\")
else()
    set(CMAKE_PREFIX_PATH "/usr/local/Qt/5.7/gcc_64")
    if(EXISTS "/usr/local/boost_1_68_g++_8.1.0/")
        set(BOOST_ROOT "/usr/local/boost_1_68_g++_8.1.0/")
    else()
        message( FATAL_ERROR "Boost not found" )
    endif()
endif()

#### Find boost static and shared libraries ####
# if (UNIX)
#   set(Boost_USE_STATIC_LIBS OFF)
#   find_package(Boost REQUIRED COMPONENTS system filesystem serialization iostreams thread date_time program_options unit_test_framework bzip2 zlib chrono regex)
#   set(BOOST_SHARED_LIBRARIES ${Boost_LIBRARIES})
#   unset(Boost_LIBRARIES)
# endif()

set(Boost_USE_STATIC_LIBS ON)
if (WIN32)
find_package(Boost REQUIRED COMPONENTS system filesystem serialization iostreams thread date_time program_options unit_test_framework chrono regex bzip2 zlib)
  set(BOOST_STATIC_LIBRARIES ${Boost_LIBRARIES})
else()
  find_package(Boost REQUIRED COMPONENTS system filesystem serialization iostreams thread date_time program_options unit_test_framework chrono regex)
  find_package(ZLIB REQUIRED)
  find_package(BZip2 REQUIRED)
  set(BOOST_STATIC_LIBRARIES ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES})
endif()
unset(Boost_LIBRARIES)
message( "Found boost at ${Boost_LIBRARY_DIRS}" )

#### Find Qt dependencies ####
find_package(Qt5 REQUIRED COMPONENTS Core Charts Widgets Gui PrintSupport Multimedia)

#### Protobuf libraries ####
if (WIN32)
    set(PROTOBUF_DEBUG_LIBRARIES "C:/Sys/protobuf-2.6.1/vsprojects/x64/Debug/libprotobuf.lib")
    set(PROTOBUF_RELEASE_LIBRARIES "C:/Sys/protobuf-2.6.1/vsprojects/x64/Release/libprotobuf.lib")
    set(PROTOBUF_HEADERS "C:/Sys/protobuf-2.6.1/vsprojects/include")
else()
    if (EXISTS "/usr/local/protobuf_3.6.1_g++_8.2.0/lib/libprotobuf.a")
        set(PROTOBUF_LIBRARIES "/usr/local/protobuf_3.6.1_g++_8.2.0/lib/libprotobuf.a")
        set(PROTOBUF_HEADERS "/usr/local/protobuf_3.6.1_g++_8.2.0/include/")
    else()
        message( FATAL_ERROR "protobuf not found")
    endif()
endif()

# Root directory
include_directories("${CMAKE_SOURCE_DIR}")

# NLOpt
if (WIN32)
include_directories("C:/Sys/nlopt")
endif()

#### Folly includes ####
if (EXISTS "/usr/local/folly")
include_directories("/usr/local/folly")
endif()

#### Boost includes ####
if (Boost_FOUND)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    add_definitions( "-DHAS_BOOST" )
endif()

if (EXISTS "${CMAKE_SOURCE_DIR}/D_Env_Live_OnixS/FIX8/include")
    include_directories(D_Env_Live_OnixS/FIX8/include)
endif()

if (EXISTS "${CMAKE_SOURCE_DIR}/A_Utils/Snappy")
    include_directories(A_Utils/Snappy)
    include_directories(${Snappy_BINARY_DIR})
endif()

if (UNIX)
    include_directories(
        /usr/local/lib/R/site-library/Rcpp/include
        /usr/share/R/include)
endif()

set(BUILDING_OTHER_PROJECTS true)

MACRO(sub_dir_list result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF((IS_DIRECTORY "${curdir}/${child}") AND (EXISTS "${curdir}/${child}/CMakeLists.txt"))
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

sub_dir_list(SUBDIRS ${CMAKE_SOURCE_DIR})

#Versioning
#configure_file (
#  "${PROJECT_SOURCE_DIR}/version.h.in"
#  "${PROJECT_BINARY_DIR}/version.h"
#)

include_directories("${PROJECT_BINARY_DIR}")

FOREACH(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
ENDFOREACH()
