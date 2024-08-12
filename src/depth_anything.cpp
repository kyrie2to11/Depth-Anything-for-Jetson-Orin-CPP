#include "depth_anything.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>

//----------------------------------------------------------------------------------------
DepthAnything::DepthAnything(const std::string& engine_file_path)
{   
    // 1. 加载 engine 到内存中
    std::ifstream file(engine_file_path, std::ios::binary); // 创建一个 std::ifstream (input file stream)对象，以便读取 engine_file_path 文件
    assert(file.good()); // 检查文件是否处于有效状态
    file.seekg(0, std::ios::end); // 移动文件指针到文件末尾
    auto size = file.tellg(); // 获取文件大小
    file.seekg(0, std::ios::beg); // 移动文件指针到文件开头
    char* trtModelStream = new char[size]; // 为 engine 存放模型数据分配内存
    assert(trtModelStream); // 检查分配是否成功
    file.read(trtModelStream, size); // 读取 engine 存放模型数据到 trtModelStream
    file.close(); // 关闭文件

    // 2. 创建运行时 runtime
    initLibNvInferPlugins(&this->gLogger, ""); // 初始化 libnvinfer_plugin 库，使用 gLogger 作为日志记录器，并且插件命名空间为空
    this->gLogger.log(nvinfer1::ILogger::Severity::kINFO, "DepthAnything engine file loaded"); // 打印日志信息
    this->runtime = nvinfer1::createInferRuntime(this->gLogger); // 创建一个 nvinfer1::IRuntime 对象，用于运行推理引擎
    assert(this->runtime != nullptr); // 检查运行时是否创建成功

    // 3. 反序列化 trtModelStream 创建 engine
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size); // 反序列化 trtModelStream 并创建 nvinfer1::ICudaEngine 对象
    assert(this->engine != nullptr); // 检查 engine 是否创建成功
    delete[] trtModelStream; // 释放 trtModelStream 内存

    // 4. 创建执行上下文 context
    this->context = this->engine->createExecutionContext(); // 创建一个 nvinfer1::IExecutionContext 对象，用于执行推理
    assert(this->context != nullptr); // 检查执行上下文是否创建成功 

    // 5. 创建 cudaStream 对象，用于管理异步操作
    cudaStreamCreate(&this->stream); 

    // 6. 获取 engine 的输入输出 binding 信息
    this->num_bindings = this->engine->getNbBindings(); // 获取 engine 的绑定数量，对应输入输出的数量
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding); // 保存输入绑定信息到 vector
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            this->num_outputs += 1;
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding); // 保存输出绑定信息到 vector
        }
    }
}
//----------------------------------------------------------------------------------------
DepthAnything::~DepthAnything()
{
    this->context->destroy(); // 销毁执行上下文
    this->engine->destroy(); // 销毁 engine
    this->runtime->destroy(); // 销毁运行时
    cudaStreamDestroy(this->stream); // 销毁 cudaStream 对象
    for (auto& ptr : this->device_ptrs) { // 循环遍历 device_ptrs 的元素
        CHECK(cudaFree(ptr)); // 释放 device_ptrs 内存，并检查是否成功
    }

    for (auto& ptr : this->host_ptrs) { // 循环遍历 host_ptrs 的元素
        CHECK(cudaFreeHost(ptr)); // 释放 host_ptrs 内存，并检查是否成功
    }
}
//----------------------------------------------------------------------------------------
void DepthAnything::MakePipe(bool warmup) // 主要用来申请 device 和 host 内存, 并根据条件判断是否执行预热
{
#ifndef CUDART_VERSION // 检查 CUDART_VERSION 是否定义, 在 CUDA 工具包编译时定义
#error CUDART_VERSION Undefined!
#endif

    for (auto& bindings : this->input_bindings) { // 循环遍历输入绑定信息
        void* d_ptr;
#if(CUDART_VERSION < 11000)
        CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize)); // 申请 device 内存
#else
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream)); 
#endif
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) { // 循环遍历输出绑定信息
        void* d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
#if(CUDART_VERSION < 11000)
        CHECK(cudaMalloc(&d_ptr, size)); // 申请 device 内存
#else
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
#endif
        CHECK(cudaHostAlloc(&h_ptr, size, 0)); // 申请 host 内存
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) { // 是否进行预热
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size); // 申请 host 内存
                memset(h_ptr, 0, size); // 填充 host 内存为 0
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream)); // 异步将 host 内存拷贝到 device 内存
                free(h_ptr); // 释放 host 内存
            }
            this->Infer(); // 执行推理
        }
    }
}
//----------------------------------------------------------------------------------------
void DepthAnything::PreProcess(const cv::Mat& image, cv::Mat& out, cv::Size& size) // 图像预处理函数，将图像缩放到指定大小，并填充边框,输出用于网络输入, 并保存处理参数用于后处理
{
    const int inp_h  = size.height; // 网络输入要求的高度
    const int inp_w  = size.width;  // 网路输入要求的宽度
    int       height = image.rows;
    int       width  = image.cols;

    // resize 后 tmp 的宽高分别为 inp_w 和 inp_h
    cv::Mat tmp;
    if ((int)width != inp_w || (int)height != inp_h) {
        assert(!image.empty());
        cv::resize(image, tmp, cv::Size(inp_w, inp_h)); // 缩放图片到指定大小 size
    }
    else {
        tmp = image.clone();
    }

    // 转换为深度学习网络的输入格式：NCHW
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    /*
    `tmp` 是输入图像，

    `out` 是输出blob，

    `1 / 255.f` 是缩放因子，

    `cv::Size()` 表示不改变图像大小，

    `cv::Scalar(0, 0, 0)` 是减去的平均值，

    `true` 表示交换RB通道，

    `false` 表示不裁剪图像，

    `CV_32F` 表示输出数据类型为32位浮点数。
    */

    // normalize 输出图像 out
    out = (out - mean) / std;

    // 保存处理参数用于后处理
    this->pparam.orig_height = height;
    this->pparam.orig_width  = width;
    this->pparam.io_height   = inp_h;
    this->pparam.io_width    = inp_w;
}
//----------------------------------------------------------------------------------------
void DepthAnything::CopyFromMat(const cv::Mat& image) // 输入图像拷贝到 device 内存,缩放到模型输入要求的大小
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->PreProcess(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}
//----------------------------------------------------------------------------------------
void DepthAnything::Infer() // 从 device 内存取出数据，执行推理，并将结果拷贝到 host 内存
{
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr); // 异步执行推理， this->device_ptrs.data() 返回指向 vector 首元素的指针
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream)); // 异步将 device 内存拷贝到 host 内存
    }
    cudaStreamSynchronize(this->stream); // 等待推理完成
}
//----------------------------------------------------------------------------------------
void DepthAnything::PostProcess(cv::Mat& color_map)
{
    auto&     orig_h = this->pparam.orig_height;
    auto&     orig_w  = this->pparam.orig_width;
    auto&     out_h = this->pparam.io_height;
    auto&     out_w  = this->pparam.io_width;

    // 取出推理结果并转化为一个 CV_32FC1 Mat, 并映射到 0-255 范围
    cv::Mat depth_pred(out_h, out_w, CV_32FC1, this->host_ptrs[0]);
    cv::normalize(depth_pred, depth_pred, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Resize 输出结果到原始图像大小
    if (orig_h != out_h || orig_w != out_w) {
        cv::resize(depth_pred, depth_pred, cv::Size(orig_w, orig_h)); 
    }

    // 保存结果到 color_map
    cv::applyColorMap(depth_pred, color_map, cv::COLORMAP_INFERNO);
}
//----------------------------------------------------------------------------------------