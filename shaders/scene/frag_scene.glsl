#version 450 core

// todo: define all the input variables to the fragment shader
in vec4 fragPosLightSpace;
in vec3 frag_normal; // Received world-normal from vertex shader
in vec3 fragPos;

// todo: define all the uniforms
uniform vec3 material_color;
uniform vec3 light_pos;

layout(binding = 0) uniform sampler2D depthTex;  // depth texture bound to texture unit 0
out vec4 outColor;

void main() {
    // todo: fill in the fragment shader

    vec3 light_dir = normalize(light_pos.xyz - fragPos);

    vec3 normalized_normal = normalize(frag_normal);

    vec3 fragPos3D = fragPosLightSpace.xyz / fragPosLightSpace.w;
    fragPos3D = (fragPos3D + 1.0) / 2.0;

    float z_current = fragPos3D.z;
    float z_depthTex = texture(depthTex, fragPos3D.xy).r;

    vec3 diffuse_color = material_color * clamp(dot(normalized_normal, light_dir), 0, 1);

    float bias = max(0.0005 * (1.0 - dot(frag_normal, light_dir)), 0.0001);

    if(z_current - bias > z_depthTex) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {

        outColor = vec4(diffuse_color, 1.0);
    }
}