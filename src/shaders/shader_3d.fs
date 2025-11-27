#version 330 core
out vec4 FragColor;

in float vIntensity;

const vec3 kPhases = vec3(5.423, 4.359, 1.150);

void main()
{
    // if (vIntensity < 990.0)
    // {
        // FragColor.rgb = sin(vec3(vIntensity) * 0.05 + kPhases) * 0.5 + 0.5;
        FragColor.r = 0.7 * 0.7 + vIntensity * 0.0005;
        FragColor.g = 0.5 * 0.7;
        FragColor.b = 1.0 * 0.7 - vIntensity * 0.0005;
        FragColor.a = 1.0;
    // }
    // else
    // {
    //     FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    // }
}