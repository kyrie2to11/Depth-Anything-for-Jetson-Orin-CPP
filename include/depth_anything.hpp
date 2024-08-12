#ifndef DEPTH_ANYTHING_HPP
#define DEPTH_ANYTHING_HPP
#include "NvInferPlugin.h"
#include "utils.hpp"
#include <fstream>

using namespace depth_anything;

class DepthAnything {
private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR}; // Logger 对象作为 DepthAnything 的成员变量，随 DepthAnything 对象创建而创建，随 DepthAnything 对象销毁而销毁
public:
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs; // void* 用于存储指向任意类型的指针
    cv::Scalar           mean = cv::Scalar(0.485, 0.456, 0.406); // 图像预处理时对图像进行归一化的均值
    cv::Scalar           std  = cv::Scalar(0.229, 0.224, 0.225); // 图像预处理时对图像进行归一化的方差

    PreParam             pparam;

public:
    explicit DepthAnything(const std::string& engine_file_path); // 构造函数
    ~DepthAnything(); // 析构函数

    void                 MakePipe(bool warmup = true);
    void                 CopyFromMat(const cv::Mat& image);
    void                 PreProcess(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 Infer();
    void                 PostProcess(cv::Mat& color_map);
};
#endif  // DEPTH_ANYTHING_HPP

