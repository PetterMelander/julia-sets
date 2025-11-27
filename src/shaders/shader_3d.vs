#version 330 core
layout (location = 0) in vec2 aPos;

uniform mat4 lookAt;
uniform sampler2D texture2;
out float vIntensity;

void main()
{
    vec2 uv = aPos * 0.5 + 0.5;

    float h = texture(texture2, uv).r;

    vec4 pos = vec4(aPos.x, sqrt(sqrt(h)) * 0.05, aPos.y, 1.0);

    gl_Position = lookAt * pos;
    vIntensity = h;
}