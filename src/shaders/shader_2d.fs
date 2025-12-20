#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;

const vec3 kPhases = vec3(5.423, 4.359, 1.150);

void main()
{
    float intensity = texture(texture1, TexCoord).r;
    if (intensity < 2500.0)
    {
        FragColor.rgb = sin(vec3(intensity * 0.05) + kPhases) * 0.5 + 0.5;
        FragColor.a = 1.0;
    }
    else
    {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}