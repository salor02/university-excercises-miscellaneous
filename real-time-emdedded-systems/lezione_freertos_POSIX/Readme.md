# FreeRTOS LINUX/POSIX Port

## Introduction
This workspace allows you to compile FREERTOS applications in a Linux/POSIX environment.

## NOTE
This branch `noros` does not include microROS library or examples. If you are interested in working with microROS switch to the branch corresponding to your ROS2 distro.


## Install Prerequisite
1. Install cmake and compilers to build this repo:
```
$ sudo apt-get install build-essential
```

## Building Examples
Recursively clone the noros branch
```
$ git clone --recursive -b noros https://github.com/cscribano/uros_freertos_posix.git
```

Compile all the sample applications provided:
```
$ cd uros_freertos_posix
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## References
This package is released for teaching and educational purposes only.