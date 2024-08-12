# Prerequisites

## Hardware

- Jetson Orin NX
- Intel Realsense D435i

## Software

- JetPack 5.1.1
- OpenCV 4.5.4
- CUDA 10.2
- cuDNN 8.0.0
- TensorRT 7.1.3.4
- CMake 3.10.2
- Python 3.6.9
- PyTorch 1.6.0
- TorchVision 0.7.0
- NumPy 1.19.1
- Matplotlib 3.2.1
- Pillow 7.2.0
- PyYAML 5.3.1
- Cython 0.29.21
- Pytest 6.0.1
- Pylint 2.5.3
- Sphinx 3.2.1
- Sphinx-rtd-theme 0.5.0 

# Building the Project

## Cloning the Repository

``` bash
git clone https://github.com/jetsonhacks/Depth-Anything-for-Jetson-Orin-CPP.git
```

## Build the tensor RT engine

``` bash
cd Depth-Anything-for-Jetson-Orin-CPP/tensorrt_engine
./build_engine.sh
```

## Building the Libraries

``` bash
cd Depth-Anything-for-Jetson-Orin-CPP
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Running the Demo

``` bash
cd Depth-Anything-for-Jetson-Orin-CPP/build/
./depth_anything_demo
``` 

# Running the Tests

``` bash
cd Depth-Anything-for-Jetson-Orin-CPP/build/
make test
``` 

# Results

