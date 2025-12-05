#version 330 core
layout (location = 0) in vec2 aPos;

uniform mat4 lightSpaceMatrix;
uniform sampler2D heightMap;

void main()
{
    vec2 uv = aPos * 0.5 + 0.5;
    float h = texture(heightMap, uv).r;

    vec4 pos = vec4(aPos.x, sqrt(sqrt(h)) * 0.075, aPos.y, 1.0);
    gl_Position = lightSpaceMatrix * pos;
}