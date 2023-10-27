#include <iostream>
#include <opencv2/opencv.hpp>

#include <GL/glew.h>
#include <GL/glut.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

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

ImageShader compileImageShader( const char* vertexShaderSource, const char* fragmentShaderSource, cv::Mat image) {
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
        1.0f,  1.0f, 0.0f,  1.0f, 1.0f,
        1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f
    };
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.cols, image.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data );
    glBindTexture(GL_TEXTURE_2D, 0);

    return { shaderProgram, VAO, texture };
}

void useImageShader( ImageShader imageShader, cv::Mat image) {
    glUseProgram(imageShader.shaderProgram);
    glBindVertexArray(imageShader.VAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, imageShader.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.cols, image.rows, GL_RGBA, GL_UNSIGNED_BYTE, image.data );
    glUniform1i(glGetUniformLocation(imageShader.shaderProgram, "texture1"), 0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

int main( int argc, char* argv[] )
{
    RunMPPGraph("/home/xallt/progs/HandTrackingProject/cpp_app/dependencies/mediapipe/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu.pbtxt", "", "");
    exit(0);

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
    cv::cvtColor( frame, frame, cv::COLOR_BGR2RGBA );

    GLFWwindow* window = glfwCreateWindow( frame.cols, frame.rows, "glfw window", nullptr, nullptr );
    glfwSetWindowCloseCallback( window, []( GLFWwindow* window ){ glfwSetWindowShouldClose( window, GL_FALSE ); } );
    glfwMakeContextCurrent( window );
    glfwSetWindowSizeLimits( window, frame.cols, frame.rows, frame.cols, frame.rows);
    glfwSwapInterval( 1 );

	glewExperimental = GL_TRUE; // stops glew crashing on OSX :-/
    if (glewInit() != GLEW_OK) {
        std::cout << "Could not initialize GLEW" << std::endl;
        return -1;
    }

    bool firstIteration = true;
    ImageShader imageShader = compileImageShader(vertexShaderSource, fragmentShaderSource, frame);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL( window, true );
    ImGui_ImplOpenGL3_Init( "#version 330" );


    bool is_show = true;
    bool flagFlipHorizontally = true;
    uint32_t width, height;
    while (is_show) {
        glfwPollEvents();
        cap.read(frame);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);

        // Flip vertically and horizontally
        cv::flip(frame, frame, 0);
        if (flagFlipHorizontally)
            cv::flip(frame, frame, 1);

        glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
        glClear( GL_COLOR_BUFFER_BIT );

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (firstIteration) {
            // Align to upper right corner
            ImGui::SetNextWindowPos( ImVec2( frame.cols, 0 ), 0, ImVec2(1,0) );
            // Set transparent background
            ImGui::SetNextWindowBgAlpha(0.5f);
        }

        ImGui::Begin( "ImGUI controls", &is_show );
        ImGui::Checkbox( "Flip horizontally", &flagFlipHorizontally );
        

        ImGui::End();

        useImageShader( imageShader, frame);
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
        glfwSwapBuffers( window );

        firstIteration = false;
        // Check if q was pressed
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            is_show = false;
        }
    }

    ImGui_ImplGlfw_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    return 0;
}