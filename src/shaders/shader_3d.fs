#version 330 core
out vec4 FragColor;

in float vIntensity;
in vec3 vNorm;
in vec3 fragPos;
in vec4 fragPosLightSpace;

uniform vec3 viewPos;
uniform sampler2D shadowMap;

const vec3 lightColor = vec3(0.5, 0.5, 0.5);
const vec3 lightDir = vec3(-0.4472135955, 0.894427191, 0.0);
const vec3 ambient = vec3(0.1, 0.1, 0.1);
const float specularStrength = 0.5;

const vec3 gammaInv = vec3(1.0/2.2);

float shadowCalculation(vec4 fragPosLightSpace, vec3 norm)
{
    vec3 projCoords = fragPosLightSpace.xyz * 0.5 + 0.5;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    // float bias = max(0.005 * (1.0 - dot(norm, lightDir)), 0.0005);
    float bias = 0.005;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    // float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}

void main()
{
    vec3 color = vec3(0.5 + vIntensity * 0.0015, 0.7, 0.9 - vIntensity * 0.0005);

    vec3 norm = normalize(vNorm);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 64);
    vec3 specular = specularStrength * spec * lightColor;

    float shadow = shadowCalculation(fragPosLightSpace, norm);
    color = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;
    FragColor = vec4(pow(color, gammaInv), 1.0);
}