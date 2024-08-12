#include "chrono"
#include "opencv2/opencv.hpp"
#include "depth_anything.hpp"

using namespace std;

// 定义函数将 RS2 帧转换为 OpenCV Mat
cv::Mat frame_to_mat(const rs2::frame& frame) {
    rs2::video_frame video_frame = frame.as<rs2::video_frame>();
    const int width = video_frame.get_width();
    const int height = video_frame.get_height();
    const int bpp = video_frame.get_bytes_per_pixel();
    return cv::Mat(cv::Size(width, height), CV_8UC(bpp), (void*)video_frame.get_data(), cv::Mat::AUTO_STEP);
}


int main(int argc, char** argv)
{
    float    f, total_f;
    float    FPS[16], TotalTime[16];
    int      i, Fcnt=0;
    cv::Mat  image;
    cv::Mat combined_image; // 用于获取水平拼接 color_map 和 image 的结果
    std::chrono::steady_clock::time_point TotalBegin, TotalEnd, InferBegin, InferEnd;
    bool use_stream = false;
    bool use_image = false;
    bool use_video = false;

    // 要求输入4个参数
    if (argc < 3) {
        fprintf(stderr,"Usage: ./build/depth_anything_cpp [model_trt.engine] [--stream or --image or --video] [camera_id or image_path or video path] \n");
        return -1;
    }

    // 解析命令行参数
    if (strcmp(argv[2],"--stream") == 0) {
        cout << "Streaming from camera..." << endl;
        use_stream = true;
    } else if (strcmp(argv[2],"--image") == 0) {
        cout << "Loading image from path: " << argv[3] << "..." << endl;
        use_image = true;
    } else if (strcmp(argv[2],"--video") == 0) {
        cout << "Loading video from path: " << argv[3] << endl;
        use_video = true;
    } else {
        cout << "Invalid argument: " << argv[2] << endl;
        return -1;
    }

    const string engine_file_path = argv[1];
    const string path = argv[3];


    for (i=0;i<16;i++) FPS[i]=0.0;

    cout << "\nSet CUDA...\n" << endl;

    cudaSetDevice(0);

    cout << "Loading TensorRT model from: " << engine_file_path << endl;
    cout << "\nWait a second..." << std::flush; // 刷新缓冲区，使输出立即显示
    auto depth_anything = new DepthAnything(engine_file_path);

    cout << "\rLoading the pipe... " << string(10, ' ')<< endl ; // \r 是一个控制字符，表示将光标移动到当前行的行首，但不换行
    depth_anything->MakePipe(true);
    cout << "\nPipe is ready." << endl;

    // 用于加载视频
    cv::VideoCapture cap;

    // 用于 realsense 捕获视频流
    // 创建rs2::context对象，用于管理和控制相机
    rs2::context ctx;

    // 获取已连接的RealSense设备列表
    auto devices = ctx.query_devices();
    rs2::pipeline pipe;
    rs2::config cfg;
    int device_num = int(*argv[3] - '0');
    cfg.enable_device(devices[device_num].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);


    if (use_video) {
        cap.open(path);
        if (!cap.isOpened()) {
            cerr << "ERROR: Unable to open the video file" << endl;
            return -1;
        }
    } else if (use_stream) {
        pipe.start(cfg);
    }


    while (1) {
        if(use_video){
            cap >> image; 
            if (image.empty()) {
                cap.open(path); // 到达视频末尾重新打开视频文件
                continue;
            }
        } else if (use_image) {
            image = cv::imread(path);
        } else {
            // 等待深度相机捕捉到帧
            rs2::frameset frameset = pipe.wait_for_frames();
            rs2::frame color_frame = frameset.get_color_frame();

            // 将RS2帧转换为OpenCV Mat
            image = frame_to_mat(color_frame);
        }
 
        // 预处理
        TotalBegin = std::chrono::steady_clock::now();
        depth_anything->CopyFromMat(image);

        // 推理
        InferBegin = std::chrono::steady_clock::now();
        depth_anything->Infer();
        InferEnd = std::chrono::steady_clock::now();

        // 后处理
        cv::Mat color_map;
        depth_anything->PostProcess(color_map);
        TotalEnd = std::chrono::steady_clock::now();
        // 计算帧率并放入 color_map
        f = std::chrono::duration_cast <std::chrono::milliseconds> (InferEnd - InferBegin).count(); // 模板函数 std::chrono::duration_cast 用于将 std::chrono::steady_clock::time_point 类型转换为指定类型的时间值，这里将其转换为毫秒
        total_f = std::chrono::duration_cast <std::chrono::milliseconds> (TotalEnd - TotalBegin).count(); // 计算总时间
        if (f>0.0) {
            FPS[(Fcnt&0x0F)]=1000.0/f; // (Fcnt++)&0x0F：将 Fcnt 与 0x0F 进行位运算，实现循环队列，每16次循环计算一次平均帧率
            TotalTime[(Fcnt&0x0F)]=total_f; 
            Fcnt++;
        }
        for (f=0.0, total_f=0.0, i=0; i<16; i++) { 
            f += FPS[i];
            total_f += TotalTime[i];
        }
        cv::putText(color_map, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255)); // 显示帧率, 参数依次是：图像、文字、坐标、字体、大小、颜色
        cv::putText(color_map, cv::format("InferTime %0.2f ms", 1000.0/f*16),cv::Point(10,45),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255)); // 显示推理时间
        cv::putText(color_map, cv::format("TotalTime %0.2f ms", total_f/16),cv::Point(10,70),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255)); // 显示总时间

        // 可视化输出 color_map 和 image 水平拼接结果
        cv::hconcat(image, color_map, combined_image);
        cv::imshow("Depth Anything on Jetson Orin NX - 16GB RAM", combined_image);
        char key = cv::waitKey(1);
        if(key == 'q' || key == 27) break;
    }

    // 释放资源
    cap.release();
    pipe.stop();
    cv::destroyAllWindows();
    delete depth_anything;

    return 0;
}