#version 330 core
out vec4 FragColor;

uniform float increment;

void main()
{
    FragColor = vec4(increment, increment, increment, 1.0);
}