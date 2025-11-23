#version 330 core
layout (location = 0) in vec2 aPos;

uniform sampler2D heightMap;
out float vIntensity;

void main()
{
    vec2 uv = aPos * 0.5 + 0.5;

    vec2 texelSize = 1.0 / vec2(textureSize(heightMap, 0));

    float h = texture(heightMap, uv).r;

    float zoom = 1.0;
    float angle = -1.0; 

    vec3 pos = vec3(aPos.x, h, aPos.y);

    float c = cos(angle);
    float s = sin(angle);
    float newY = pos.y * c - pos.z * s;
    float newZ = pos.y * s + pos.z * c;

    gl_Position = vec4(pos.x * zoom, newY * zoom, newZ * zoom, 1.0);
    vIntensity = h;
}