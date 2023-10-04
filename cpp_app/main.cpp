#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error: Could not access webcam" << std::endl;
        return -1;
    }

    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    
    while (1) {
        cap >> frame;
        cv::imshow("Webcam", frame);
        if (cv::waitKey(1) >= 0) break;
    }

    cap.release();
    cv::destroyWindow("Webcam");
    return 0;
}
