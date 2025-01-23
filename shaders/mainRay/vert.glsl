#version 420 core

// Input attributes
in vec3 position;    // we can also use layout to specify the location of the attribute
in vec2 uv;
in vec3 normal;

// Uniform variables
uniform float scale;
uniform vec3 center;
uniform float aspect;
uniform mat4 model_matrix;
uniform mat4 projection_matrix;
uniform mat4 view_matrix;

// Output variable (We will use this variable to pass the normal to the fragment shader)
// out vec3 fragNormal;
out vec3 fragNormal;

void main() 
{
	// Transform the position from object space (a.k.a model space) to clip space. 
	// The range of clip space is [-1,1] in all 3 dimensions.
	// vec4 pos = model_matrix * vec4(position, 1.0);
	vec4 pos = projection_matrix * view_matrix *  model_matrix * vec4(position, 1.0);
	
	// pos.x /= pos.x;			// Correction for aspect ratio (optional)
	// pos.z *= -1;			// Negate pos.z to make rayman face forward 
		
	
	gl_Position = pos;

	// Transform the normal from object (or model) space to world space.
	mat4 normal_matrix = transpose(inverse(model_matrix));
	vec3 new_normal = (normal_matrix*vec4(normal,0)).xyz;
	fragNormal = normalize(new_normal);

	// fragUV = uv;
}