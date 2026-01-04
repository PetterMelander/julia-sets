#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D mandelbrot;

void main()
{
    float tex = texture(mandelbrot, TexCoord).r;
    FragColor = vec4(tex, tex, tex, 1.0);
}