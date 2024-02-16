# Hand Tracking App with Mediapipe and CMake

<video src="assets/demo_fingers.mp4" width="320" height="240" controls></video>

## How to Run the CMake Project

```bash
mkdir build
cd build
cmake ..
make
./main
```

For a detailed setup and description of the project, please explore the [detailed approach post](https://xallt.github.io/posts/connecting-mediapipe-cmake/).

## Repository Overview
- Integration of Mediapipe (based on Bazel) into a CMake project
- Hand Tracking demo
- Dear ImGUI interface for basic visualization controls

## Challenges Faced
- **Mediapipe and Bazel:** Integrating Mediapipe, which primarily uses the Bazel build system, into a CMake project required extensive tinkering to bridge the gap.
- **Python API Limitations:** CPU-only inference in Python and the interpreter's general slowness posed challenges.
- **Transition to C++:** Initial implementation in Python using Mediapipe Python API, transitioning to C++ for improved performance.
- **Using Mediapipe as a Library:** Setup of Mediapipe's Hand Tracking as a library, requiring source code modifications and integration with CMake for effective usage.

## Repository Structure
- [HandTrackingProject](https://github.com/Xallt/HandTrackingProject): CMake project for the app
- [Mediapipe](https://github.com/Xallt/mediapipe): Fork of Mediapipe as a submodule

## Acknowledgments
- [Mediapipe](https://github.com/google/mediapipe): Core framework for hand tracking
- [LibMP](https://github.com/jiangzhihao/libmp): Inspirational project for setup
