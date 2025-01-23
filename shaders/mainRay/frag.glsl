#version 420 core

// Input variables, i.e., the output variable from the vertex shader
in vec3 fragNormal;
//in vec2 fragUV;

layout (binding=0) uniform sampler2D tex2D;

// Output variables
out vec4 outColor;

void main() {
	// Normalize the normal and use it to compute the color of the fragment:
	// You may simply take the absolute value of the normal, OR
	// add 1.0 to each component of the normal, and divide each component by 2.
	vec3 norm = abs(normalize(fragNormal));

	// Assign the color to "outColor".
	outColor = vec4(norm, 1.0);

	// vec3 texColor = texture(tex2D,fragUV).rgb;
    // outColor = vec4(texColor, 1.0);
}