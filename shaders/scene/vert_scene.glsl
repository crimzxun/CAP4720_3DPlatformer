#version 450 core

// Attributes
layout(location = 0) in vec3 position;    // we can also use layout to specify the location of the attribute
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;

uniform mat4 lightProjectionMatrix;
uniform mat4 lightViewMatrix;
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

// todo: define all the out variables
out vec4 fragPosLightSpace;
out vec3 fragPos; // Pass world position to fragment shader
out vec3 frag_normal; // Pass world-normal to fragment shader

// todo: define all the uniforms

void main() {
    // todo: fill in vertex shader
    fragPos = (modelMatrix * vec4(position, 1.0)).xyz;

    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);

    frag_normal = (transpose(inverse(modelMatrix)) * vec4(normal, 0)).xyz;

    fragPosLightSpace = lightProjectionMatrix * lightViewMatrix * modelMatrix * vec4(position, 1.0);
}