#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main() {
    cout << "I want you to give me an image path and I'll read the image for you" << endl;
    string s;
    cin >> s;
    cv::Mat img = cv::imread(s);
    if (img.empty()) {
        std::cerr << "Error: Could not read the image" << std::endl;
        return 1;
    }
    // Image read successfully. Do nothing.
    return 0;
}
