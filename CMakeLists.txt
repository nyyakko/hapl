cmake_minimum_required(VERSION 3.25)

project(hapl LANGUAGES CXX)

include(cmake/static_analyzers.cmake)
include(cmake/get_cpm.cmake)

# set(hapl_CompilerOptions ${hapl_CompilerOptions})
# set(hapl_LinkerOptions ${hapl_LinkerOptions})

CPMAddPackage("gl:libeigen/eigen#3.4.0")
CPMAddPackage("gh:fmtlib/fmt#10.2.1")

set(hapl_ExternalLibraries Eigen3::Eigen fmt::fmt)

add_subdirectory(hapl)

