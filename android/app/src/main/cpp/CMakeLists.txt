cmake_minimum_required(VERSION 3.22.1)
project(mylib)

add_library(mylib SHARED
  mylib.cpp
)

target_include_directories(mylib PRIVATE
  .
)
