#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aNorm;

uniform mat4 lookAt;
uniform sampler2D texture2;
uniform float xstep;
uniform float ystep;

out float vIntensity;
out vec3 vNorm;
out vec3 fragPos;

void main()
{
    vec2 uv = aPos * 0.5 + 0.5;

    float h = texture(texture2, uv).r;

    vec4 pos = vec4(aPos.x, sqrt(sqrt(h)) * 0.075, aPos.y, 1.0);

    vNorm = vec3(aNorm.x * ystep, 6 * xstep * ystep, aNorm.y * xstep);

    gl_Position = lookAt * pos;
    vIntensity = h;
    vNorm = vNorm;
    fragPos = vec3(pos);
}