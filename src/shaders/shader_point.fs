#version 330 core
out vec4 FragColor;

uniform float speed;

void main()
{
    float scale = clamp(speed, 0.0, 1.0);
    scale = pow(scale, 4);
    vec3 color = normalize(vec3(scale, 0.75 - scale, 1.0 - scale));
    FragColor = vec4(color, 1.0);
}