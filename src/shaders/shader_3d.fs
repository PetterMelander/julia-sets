#version 330 core
out vec4 FragColor;

in float vIntensity;

void main()
{
    FragColor = vec4(0.5 - sin(vIntensity * 5 + 0.5), 0.5 - sin(vIntensity * 5 + 1.0), 0.5 - sin(vIntensity * 5 + 2.0), 1.0);
}