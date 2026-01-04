#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Defines several possible options for camera movement. Used as abstraction to
// stay away from window-system specific input methods
enum Camera_Movement
{
  FORWARD,
  BACKWARD,
  LEFT,
  RIGHT
};

// Default camera values
constexpr float YAW = -90.0f;
constexpr float PITCH = 45.0f;
constexpr float SPEED = 100.0f;
constexpr float ZOOM = 1.0f;

// An abstract camera class that processes input and calculates the
// corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera
{
public:
  glm::vec3 front;
  glm::vec3 up;
  glm::mat4 projection;

  float yaw;
  float pitch;

  float movementSpeed;
  float zoom;

  Camera(int width, int height, float yaw = YAW, float pitch = PITCH)
      : front(glm::vec3(0.0f, 0.0f, -1.0f)), yaw(yaw), pitch(pitch), movementSpeed(SPEED), zoom(ZOOM)
  {
    projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
    updateCameraVectors();
  }

  glm::mat4 GetTransform()
  {
    return projection * glm::lookAt(front, glm::vec3(0.0f, 0.0f, 0.0f), up);
  }

  // processes input received from any keyboard-like input system. Accepts input
  // parameter in the form of camera defined ENUM (to abstract it from windowing
  // systems)
  void ProcessKeyboard(Camera_Movement direction, float deltaTime)
  {
    float velocity = movementSpeed * deltaTime;
    if (direction == FORWARD)
      pitch += velocity;
    if (direction == BACKWARD)
      pitch -= velocity;
    if (direction == LEFT)
      yaw += velocity;
    if (direction == RIGHT)
      yaw -= velocity;

    if (pitch > 89.0f)
      pitch = 89.0f;
    if (pitch < -89.0f)
      pitch = -89.0f;

    updateCameraVectors();
  }

private:
  void updateCameraVectors()
  {
    // calculate the new Front vector
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront) * zoom * 1.5f;
    // also re-calculate the up vector
    glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    up = glm::normalize(glm::cross(right, front));
  }
};
