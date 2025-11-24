#version 330 core
layout (location = 0) in vec2 aPos;

uniform mat4 lookAt;
uniform sampler2D heightMap;
out float vIntensity;

void main()
{
    vec2 uv = aPos * 0.5 + 0.5;

    vec2 texelSize = 1.0 / vec2(textureSize(heightMap, 0));

    float h = texture(heightMap, uv).r;

    vec4 pos = vec4(aPos.x, h, aPos.y, 1.0);

    gl_Position = lookAt * pos;
    vIntensity = h;
}