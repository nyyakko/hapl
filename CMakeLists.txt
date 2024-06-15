cmake_minimum_required(VERSION 3.25)

project(hapl LANGUAGES CXX)

include(cmake/static_analyzers.cmake)
include(cmake/get_cpm.cmake)

# set(hapl_CompilerOptions ${hapl_CompilerOptions})
# set(hapl_LinkerOptions ${hapl_LinkerOptions})

CPMAddPackage("gh:Dobiasd/FunctionalPlus#v0.2.24")
CPMAddPackage("gh:fmtlib/fmt#10.2.1")
CPMAddPackage("gl:libeigen/eigen#3.4.0")
CPMAddPackage(
    NAME raylib
    URL "https://github.com/raysan5/raylib/archive/refs/tags/5.0.zip"
    GIT_TAG 5.0
    VERSION 5.0
    GIT_SHALLOW ON
    OPTIONS
        "WITH_PIC TRUE"
        "BUILD_SHARED_LIBS TRUE"
)

set(hapl_ExternalLibraries FunctionalPlus::fplus fmt::fmt raylib Eigen3::Eigen)

add_subdirectory(hapl)

