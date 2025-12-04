#version 330 core
out vec4 FragColor;

in float vIntensity;
in vec3 vNorm;
in vec3 fragPos;

uniform vec3 viewPos;

// const vec3 lightPos = vec3(0.5, 1.0, 0.0);
const vec3 lightColor = vec3(0.5, 0.5, 0.5);
const vec3 lightDir = vec3(-0.4472135955, 0.894427191, 0.0);
const vec3 ambient = vec3(0.1, 0.1, 0.1);
const float specularStrength = 0.5;

const vec3 gammaInv = vec3(1.0/2.2);

void main()
{
    vec3 color = vec3(0.5 + vIntensity * 0.0015, 0.7, 0.9 - vIntensity * 0.0005);

    vec3 norm = normalize(vNorm);
    // vec3 lightDir = normalize(fragPos - lightPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(viewPos - fragPos);
    // vec3 reflectDir = reflect(lightDir, -norm);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 64);
    vec3 specular = specularStrength * spec * lightColor;

    color = (ambient + diffuse + specular) * color;
    FragColor = vec4(pow(color, gammaInv), 1.0);
}