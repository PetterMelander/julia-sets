#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aNorm;

uniform mat4 lookAt;
uniform sampler2D heightMap;
uniform float xstep; // TODO: make constant?
uniform float ystep; // TODO: make constant?
uniform mat4 lightSpaceMatrix;

out float vIntensity;
out vec3 vNorm;
out vec3 fragPos;
out vec4 fragPosLightSpace;
out vec2 uv;

void main()
{
    uv = aPos * 0.5 + 0.5;
    float h = texture(heightMap, uv).r;

    vec4 pos = vec4(aPos.x, h, aPos.y, 1.0);

    vNorm = vec3(aNorm.x * ystep, 6 * xstep * ystep, aNorm.y * xstep);
    gl_Position = lookAt * pos;
    vIntensity = h;
    fragPos = vec3(pos);
    fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
}