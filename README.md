# Path Tracing in 5minutes

Simple but easily extensible single file path tracer written in C++.

This is reference implementation of [Path Tracing in 5minutes](https://speakerdeck.com/yumcyawiz/path-tracing-in-5-minutes). Choose `summer_lt_2020` tag to make codes consistent with that slide.

![](img/cornellbox.png)

## Features

* Sphere and Plane Shape
* Lambertian and Mirror BRDF
* Path Tracing

## Requirements

* C++17 or Higher
* CMake 3.12 or Higher

## Build

```
mkdir build
cd build
cmake ..
make
```

## Run

```
./build/main
```