#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <chrono>

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Error: Could not access webcam" << std::endl;
        return -1;
    }

    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;

    auto last_time = std::chrono::high_resolution_clock::now();

    while (1) {
        cap >> frame;
        auto current_time = std::chrono::high_resolution_clock::now();
        double fps = 1.0 / std::chrono::duration<double>(current_time - last_time).count();
        last_time = current_time;

        std::string fps_text = "FPS: " + std::to_string(fps);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Webcam", frame);

        if (cv::waitKey(1) >= 0) break;
    }

    cap.release();
    cv::destroyWindow("Webcam");
    return 0;
}
