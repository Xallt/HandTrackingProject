#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "run_graph_gpu.h"

const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;

void main()
{
    FragColor = texture(texture1, TexCoord);
}
)";

struct ImageShader {
    GLuint shaderProgram;
    GLuint VAO;
    GLuint texture;
};

ImageShader compileImageShader(const char* vertexShaderSource, const char* fragmentShaderSource, cv::Mat image) {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float vertices[] = {
        // positions        // texture coords
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3};

    GLuint VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.cols, image.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data);
    glBindTexture(GL_TEXTURE_2D, 0);

    return {shaderProgram, VAO, texture};
}

void useImageShader(ImageShader imageShader, cv::Mat image) {
    glUseProgram(imageShader.shaderProgram);
    glBindVertexArray(imageShader.VAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, imageShader.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.cols, image.rows, GL_RGBA, GL_UNSIGNED_BYTE, image.data);
    glUniform1i(glGetUniformLocation(imageShader.shaderProgram, "texture1"), 0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

class HandTrackingApp {
   private:
    // ImGUI variables
    bool flagFlipHorizontally = true;                                 // Mirror the image horizontally
    bool drawConnections = true;                                      // Draw lines between landmarks
    bool drawLandmarkNumbers = false;                                 // Draw landmark numbers
    bool drawSameHandTouch = false;                                   // Draw connections between landmarks on the same hand
    bool moveMouse = false;                                           // Move mouse to index finger tip
    bool drawCenter = false;                                          // Draw center of the hand
    ImVec4 landmarkColor = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);            // Color of the landmarks
    ImVec4 landmarkTouchingColor = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);    // Color of the landmarks
    ImVec4 landmarkConnectionColor = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);  // Color of the landmarks
    float touchDistance = 0.1;                                        // Distance between landmarks to draw a connection

    float last_time = 0;

   public:
    HandTrackingApp() {}
    bool isLandmarkFingertip(int landmarkIndex) {
        int connCounter = 0;
        for (int pair_num = 0; pair_num < landmarkConnections.size(); pair_num++) {
            if (landmarkConnections[pair_num].first == landmarkIndex || landmarkConnections[pair_num].second == landmarkIndex)
                connCounter++;
        }
        return connCounter == 1;
    }
    void drawLandmarks(cv::Mat& frame, const std::vector<LandmarkList>& landmarks) {
        cv::Scalar landmarkColorCV = cv::Scalar(
            landmarkColor.x * 255,
            landmarkColor.y * 255,
            landmarkColor.z * 255);
        cv::Scalar landmarkConnectionColorCV = cv::Scalar(
            landmarkConnectionColor.x * 255,
            landmarkConnectionColor.y * 255,
            landmarkConnectionColor.z * 255);
        for (int hand_num = 0; hand_num < landmarks.size(); hand_num++) {
            if (drawConnections) {
                for (int edge_num = 0; edge_num < landmarkConnections.size(); edge_num++) {
                    cv::line(
                        frame,
                        cv::Point(
                            landmarks[hand_num].landmarks[landmarkConnections[edge_num].first].x * frame.cols,
                            landmarks[hand_num].landmarks[landmarkConnections[edge_num].first].y * frame.rows),
                        cv::Point(
                            landmarks[hand_num].landmarks[landmarkConnections[edge_num].second].x * frame.cols,
                            landmarks[hand_num].landmarks[landmarkConnections[edge_num].second].y * frame.rows),
                        landmarkConnectionColorCV,
                        2);
                }
            }
            for (int i = 0; i < landmarks[hand_num].landmarks.size(); i++) {
                cv::circle(
                    frame,
                    cv::Point(
                        landmarks[hand_num].landmarks[i].x * frame.cols,
                        landmarks[hand_num].landmarks[i].y * frame.rows),
                    5, landmarkColorCV, -1);
                if (drawLandmarkNumbers) {
                    cv::putText(
                        frame,
                        std::to_string(i),
                        cv::Point(
                            landmarks[hand_num].landmarks[i].x * frame.cols + 5,
                            landmarks[hand_num].landmarks[i].y * frame.rows - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }
            }
            if (drawCenter) {
                cv::Point3f center(0.0, 0.0, 0.0);

                center += landmarks[hand_num].landmarks[HandLandmark::WRIST] / 2;
                center += landmarks[hand_num].landmarks[HandLandmark::INDEX_FINGER_MCP] / 8;
                center += landmarks[hand_num].landmarks[HandLandmark::MIDDLE_FINGER_MCP] / 8;
                center += landmarks[hand_num].landmarks[HandLandmark::RING_FINGER_MCP] / 8;
                center += landmarks[hand_num].landmarks[HandLandmark::PINKY_MCP] / 8;
                cv::circle(
                    frame,
                    cv::Point(
                        center.x * frame.cols,
                        center.y * frame.rows),
                    3, landmarkColorCV, -1
                );
            }
        }

        // Draw connections between very close landmarks
        int numLandmarksOnHand = landmarks[0].landmarks.size();
        cv::Scalar landmarkTouchingColorCV = cv::Scalar(
            landmarkTouchingColor.x * 255,
            landmarkTouchingColor.y * 255,
            landmarkTouchingColor.z * 255);
        for (int lmk_num_g = 0; lmk_num_g < numLandmarksOnHand * landmarks.size(); lmk_num_g++) {
            int hand_num = lmk_num_g / numLandmarksOnHand;
            int lmk_num = lmk_num_g % numLandmarksOnHand;
            if (!isLandmarkFingertip(lmk_num)) continue;
            for (int lmk_num2_g = lmk_num_g + 1; lmk_num2_g < numLandmarksOnHand * landmarks.size(); lmk_num2_g++) {
                int hand_num2 = lmk_num2_g / numLandmarksOnHand;
                int lmk_num2 = lmk_num2_g % numLandmarksOnHand;
                if (!isLandmarkFingertip(lmk_num2)) continue;
                if (!drawSameHandTouch && hand_num == hand_num2) continue;

                // Skip if the landmarks are on the same hand
                // And already have a connection in landmarkConnections
                if (hand_num == hand_num2) {
                    bool skip = false;
                    for (int edge_num = 0; edge_num < landmarkConnections.size(); edge_num++) {
                        if (
                            (landmarkConnections[edge_num].first == lmk_num && landmarkConnections[edge_num].second == lmk_num2) ||
                            (landmarkConnections[edge_num].first == lmk_num2 && landmarkConnections[edge_num].second == lmk_num)) {
                            skip = true;
                            break;
                        }
                    }
                    if (skip) continue;
                }

                float distance = sqrt(
                    pow(landmarks[hand_num].landmarks[lmk_num].x - landmarks[hand_num2].landmarks[lmk_num2].x, 2) +
                    pow(landmarks[hand_num].landmarks[lmk_num].y - landmarks[hand_num2].landmarks[lmk_num2].y, 2));
                if (distance < touchDistance) {
                    cv::line(
                        frame,
                        cv::Point(
                            landmarks[hand_num].landmarks[lmk_num].x * frame.cols,
                            landmarks[hand_num].landmarks[lmk_num].y * frame.rows),
                        cv::Point(
                            landmarks[hand_num2].landmarks[lmk_num2].x * frame.cols,
                            landmarks[hand_num2].landmarks[lmk_num2].y * frame.rows),
                        landmarkTouchingColorCV,
                        5);
                }
            }
        }
    }
    void drawImGui(bool& is_show) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Calculate FPS
        float current_time = glfwGetTime();
        float fps = 1.0f / (current_time - last_time);
        last_time = current_time;

        ImGui::Begin("ImGUI controls", &is_show);
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Checkbox("Flip horizontally", &flagFlipHorizontally);
        ImGui::Checkbox("Draw connections", &drawConnections);
        ImGui::Checkbox("Draw landmark numbers", &drawLandmarkNumbers);
        ImGui::Checkbox("Touching within hand", &drawSameHandTouch);
        ImGui::Checkbox("Move mouse", &moveMouse);
        ImGui::Checkbox("Draw center", &drawCenter);
        ImGui::SliderFloat("Touch distance", &touchDistance, 0.0f, 0.06f);

        ImGui::End();
    }
    int run() {
        SimpleMPPGraphRunner runner;
        runner.InitMPPGraph("/home/xallt/progs/HandTrackingProject/cpp_app/dependencies/mediapipe/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu.pbtxt");

        if (!glfwInit()) {
            std::cout << "Could not initialize GLFW" << std::endl;
            return -1;
        }

        cv::VideoCapture cap(0);

        if (!cap.isOpened()) {
            std::cout << "Error: Could not access webcam" << std::endl;
            return -1;
        }
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cout << "Error: Could not read frame from webcam" << std::endl;
            return -1;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);

        GLFWwindow* window = glfwCreateWindow(frame.cols, frame.rows, "glfw window", nullptr, nullptr);
        glfwSetWindowCloseCallback(window, [](GLFWwindow* window) { glfwSetWindowShouldClose(window, GL_FALSE); });
        glfwMakeContextCurrent(window);
        glfwSetWindowSizeLimits(window, frame.cols, frame.rows, frame.cols, frame.rows);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE;  // stops glew crashing on OSX :-/
        if (glewInit() != GLEW_OK) {
            std::cout << "Could not initialize GLEW" << std::endl;
            return -1;
        }

        ImageShader imageShader = compileImageShader(vertexShaderSource, fragmentShaderSource, frame);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        bool is_show = true;
        uint32_t width, height;
        std::vector<LandmarkList> landmarks;
        bool landmark_presence;
        float fps = 0.03f;
        last_time = glfwGetTime();
        while (is_show) {
            glfwPollEvents();
            cap.read(frame);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);

            // Flip vertically and horizontally
            if (flagFlipHorizontally)
                cv::flip(frame, frame, 1);

            // Get landmarks from Mediapipe Graph
            cv::Mat frame_copy;
            size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
            runner.ProcessFrame(frame, frame_timestamp_us, frame_copy, landmarks, landmark_presence);

            // If landmarks are present, draw them
            if (landmark_presence) {
                drawLandmarks(frame, landmarks);
            }

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            drawImGui(is_show);

            if (moveMouse && landmark_presence) {
                cv::Point3f index_tip = landmarks[0].landmarks[HandLandmark::INDEX_FINGER_TIP];
                glfwSetCursorPos(window, index_tip.x * frame.cols, index_tip.y * frame.rows);
            }

            cv::flip(frame, frame, 0);
            useImageShader(imageShader, frame);

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);

            // Check if q was pressed
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                is_show = false;
        }

        ImGui_ImplGlfw_Shutdown();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui::DestroyContext();
        glfwTerminate();

        return 0;
    }
};

int main(int argc, char* argv[]) {
    HandTrackingApp app;
    return app.run();
}