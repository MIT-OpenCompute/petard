cmake_minimum_required(VERSION 3.10)
project(petard C)

set(CMAKE_C_STANDARD 99)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/tensor.c
    src/ops.c
    src/autograd.c
)

# Create library
add_library(petard ${SOURCES})

# Example executable (if needed)
add_executable(mnist_mlp examples/mnist_mlp.c)
target_link_libraries(mnist_mlp petard m)
