#ifndef DEPTH_ANYTHING_UTILS_HPP
#define DEPTH_ANYTHING_UTILS_HPP // 防止重复包含此宏定义
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "librealsense2/rs.hpp"
#include "librealsense2/rsutil.h"

// 定义 CHECK 宏，用于检查 CUDA 内存的分配和释放是否成功
// 宏定义中允许包含两行以上命令的情形，此时必须在最右边加上”\”且该行”\”后不能再有任何字符，连注释部分都不能有
#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0) // `do { ... } while (0)` 结构确保宏在展开时作为一个独立的语句块。这种结构可以防止在某些上下文中（如 if-else 语句，防止外部 else 匹配上do while 内部的 if 条件上）出现语法错误。

class Logger: public nvinfer1::ILogger { // 继承自 nvinfer1::ILogger，实现 ILogger 接口定义的纯虚函数
public:
    nvinfer1::ILogger::Severity reportableSeverity; // 成员变量 reportableSeverity，用于设置日志输出等级

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO): // 构造函数声明，explicit 关键字用于防止隐式转换, severity 是构造函数的参数列表，默认值为 nvinfer1::ILogger::Severity::kINFO
        reportableSeverity(severity) // 构造函数执行之前，初始化成员变量 reportableSeverity
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override // 实现 ILogger 接口定义的纯虚函数 log, noexcept 表示该函数不会抛出异常， override 表示该函数是虚函数重载
    {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};

inline int get_size_by_dims(const nvinfer1::Dims& dims) // 累乘计算维度大小
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType) // 获取数据类型大小,单位 Byte
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline static float clamp(float val, float min, float max) // 限制数值范围, clamp 截断控制在 min 和 max 之间
{
    return val > min ? (val < max ? val : max) : min;
}

namespace depth_anything{
    struct Binding {
        size_t         size  = 1; // 输入或输出的大小，单位由 dsize 决定
        size_t         dsize = 1; // data type size, 单位 Byte, 如 nvinfer1::DataType::kFLOAT 对应 4 Byte
        nvinfer1::Dims dims; // 维度信息, 用于描述张量的形状
        std::string    name;
    };
    
    struct PreParam {
        // 原始输入图像大小
        int orig_height = 0;
        int orig_width  = 0;

        // 模型输入输出大小
        int io_height = 0;
        int io_width = 0;

    };
}
#endif  // DEPTH_ANYTHING_UTILS_HPP