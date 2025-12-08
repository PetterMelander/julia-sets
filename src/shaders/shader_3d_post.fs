#version 330 core
out vec4 FragColor;

in vec2 texCoords;

uniform sampler2D depthMap;
uniform sampler2D normalIntensityMap;

const vec3 gammaInv = vec3(1.0/2.2);

void main()
{
    vec3 depth = texture(normalIntensityMap, texCoords).rgb;
    // vec3 test = vec3((1.0 - depth));
    FragColor = vec4(depth, 1.0);
}