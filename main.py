import pygame as pg
from OpenGL.GL import *
import numpy as np
import shaderLoaderV3
from objLoaderV4 import ObjLoader
from utils import load_image
from guiV3 import SimpleGUI
import pyrr
import gravity
import collision
import endgame
# import timer

import menu


def upload_and_configure_attributes(object, shader):
    # VAO and VBO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, object.vertices.nbytes, object.vertices, GL_STATIC_DRAW)

    # Define the vertex attribute configurations
    # we can either query the locations of the attributes in the shader like we did in our previous assignments
    # or explicitly tell the shader that the attribute "position" corresponds to location 0.
    # It is recommended to explicitly set the locations of the attributes in the shader than querying them.
    # Position attribute
    position_loc = 0
    tex_coord_loc = 1
    normal_loc = 2
    glBindAttribLocation(shader, position_loc, "position")
    # glBindAttribLocation(shader, tex_coord_loc, "uv")
    glBindAttribLocation(shader, normal_loc, "normal")


    glVertexAttribPointer(position_loc, object.size_position, GL_FLOAT, GL_FALSE, object.stride, ctypes.c_void_p(object.offset_position))
    glVertexAttribPointer(tex_coord_loc, object.size_texture, GL_FLOAT, GL_FALSE, object.stride, ctypes.c_void_p(object.offset_texture))
    glVertexAttribPointer(normal_loc, object.size_normal, GL_FLOAT, GL_FALSE, object.stride, ctypes.c_void_p(object.offset_normal))

    glEnableVertexAttribArray(tex_coord_loc)
    glEnableVertexAttribArray(position_loc)
    glEnableVertexAttribArray(normal_loc)

    return vao, vbo, object.n_vertices


def load_cubemap_texture(filenames):
    # Generate a texture ID
    texture_id = glGenTextures(1)

    # Bind the texture as a cubemap
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    # Define texture parameters
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


    # Define the faces of the cubemap
    faces = [GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

    # Load and bind images to the corresponding faces
    for i in range(6):
        img_data, img_w, img_h = load_image(filenames[i], format="RGB", flip=False)
        glTexImage2D(faces[i], 0, GL_RGB, img_w, img_h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    # Generate mipmaps
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    # Unbind the texture
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    return texture_id



def load_2d_texture(filename):
    img_data, img_w, img_h = load_image(filename, format="RGB", flip=True)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)  # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)  # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_w, img_h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, 0)

    return texture_id


def create_framebuffer_with_depth_attachment(width, height):
    # Create a framebuffer object
    framebuffer_id = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_id)

    # Create a texture object for the depth attachment
    depthTex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depthTex_id)

    # Define texture parameters
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)  # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Attach the depth texture to the framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex_id, 0)

    # Tell OpenGL which color attachments we'll use (of this framebuffer) for rendering.
    # We won't be using any color attachments so we can tell OpenGL that we're not going to bind any color attachments.
    # So set the draw and read buffer to none
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)

    # Check if framebuffer is complete
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Framebuffer is not complete!")

    # Unbind the framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return framebuffer_id, depthTex_id




def render_scene():
    '''
    This function renders the scene from the camera's point of view.
    First implement diffuse lighting for the object and the receiver using the shaders in the scene folder.
    Then implement the shadow mapping algorithm in the same shader.
    :return:
    '''
    glUseProgram(shaderProgram_scene.shader)

   
    # todo: configure all the uniforms for the scene shader
    # example: shaderProgram_scene["viewMatrix"] = view_mat
    shaderProgram_scene["viewMatrix"] = view_mat
    shaderProgram_scene["projectionMatrix"] = projection_mat
    shaderProgram_scene["lightViewMatrix"] = light_view_mat
    shaderProgram_scene["lightProjectionMatrix"] = light_projection_mat
    shaderProgram_scene["light_pos"] = rotated_lightPos
    #shaderProgram_scene["material_color"] = material_color

    # todo: activate the texture units and bind the textures to them
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, depthTex_id)
    shaderProgram_scene["depthTex"] = 0


    # todo: make draw calls for the object:
    # send model matrix as uniform for the object, bind the vao and draw the object
    shaderProgram_scene["modelMatrix"] = model_mat_obj
    shaderProgram_scene["material_color"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    glBindVertexArray(vao_obj)
    glDrawArrays(GL_TRIANGLES, 0, obj.n_vertices) 
    # print("teapot:",model_mat_obj)
    # print("teapotx:",model_mat_obj[3][0])
    # print("teapotz:",model_mat_obj[3][2])

    # todo: make draw calls for the receiver.
    # send model matrix as uniform for the receiver, bind the vao and draw the receiver
    shaderProgram_scene["modelMatrix"] = model_mat_receiver
    shaderProgram_scene["material_color"] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    glBindVertexArray(vao_receiver)
    glDrawArrays(GL_TRIANGLES, 0, obj_receiver.n_vertices)
    # print("plane:",model_mat_receiver)


    shaderProgram_scene["modelMatrix"] = model_mat_tree
    shaderProgram_scene["material_color"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    glBindVertexArray(vao_tree)
    glDrawArrays(GL_TRIANGLES, 0, obj_tree.n_vertices)
    # print("treex:", model_mat_tree[3][0])
    # print("treey:", model_mat_tree[3][2])

    shaderProgram_scene["modelMatrix"] = model_mat_gift
    shaderProgram_scene["material_color"] = np.array([1.0, 0.843, 0.0], dtype=np.float32)
    glBindVertexArray(vao_gift)
    glDrawArrays(GL_TRIANGLES, 0, obj_gift.n_vertices)
    # print("gift:", model_mat_gift)
    # print("giftx:", model_mat_gift[3][0])
    # print("giftz:", model_mat_gift[3][2])




def render_tex():
    '''
    This function is optional. It is used to render the depth texture onto a quad for debugging purposes
    :return:
    '''
    glUseProgram(shaderProgram_visualizeTex.shader)  # being explicit even though the line below will call this function
    shaderProgram_visualizeTex["near"] = float(near)
    shaderProgram_visualizeTex["far"] = float(far)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, depthTex_id)

    glBindVertexArray(vao_receiver)
    glDrawArrays(GL_TRIANGLES, 0, obj_receiver.n_vertices)


def render_shadow_map():
    '''
    This function renders the scene from the light's point of view and stores the depth of each point in the scene
    in a texture.
    Since we don't want to render the color of the scene, we will use a custom framebuffer with a depth attachment and
    no color attachments. This depth buffer (which is also called shadow map) will be used as a texture in the next part.
    Since there is no color attachment, we won't be able to see anything on the screen.

    :return:
    '''
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_id)
    glClear(GL_DEPTH_BUFFER_BIT)

    # ***** render the object and receiver *****
    glUseProgram(shaderProgram_shadowMap.shader)  # being explicit even though the line below will call this function

    shaderProgram_shadowMap["viewMatrix"] = light_view_mat
    shaderProgram_shadowMap["projectionMatrix"] = light_projection_mat

    # ***** Draw object *****
    shaderProgram_shadowMap["modelMatrix"] = model_mat_obj
    glBindVertexArray(vao_obj)
    glDrawArrays(GL_TRIANGLES, 0, obj.n_vertices)  # Draw the triangle
    
    shaderProgram_shadowMap["modelMatrix"] = model_mat_tree
    glBindVertexArray(vao_tree)
    glDrawArrays(GL_TRIANGLES, 0, obj_tree.n_vertices)

    shaderProgram_shadowMap["modelMatrix"] = model_mat_gift
    glBindVertexArray(vao_gift)
    glDrawArrays(GL_TRIANGLES, 0, obj_tree.n_vertices)

    # ***** Draw receiver *****
    shaderProgram_shadowMap["modelMatrix"] = model_mat_receiver
    glBindVertexArray(vao_receiver)
    glDrawArrays(GL_TRIANGLES, 0, obj_receiver.n_vertices)  # Draw the triangle


    glBindFramebuffer(GL_FRAMEBUFFER, 0)



def render_skybox():
    # ******************* Draw the skybox ************************

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_id)

    # remove the translation component from the view matrix because we want the skybox to be static
    view_mat_without_translation = view_mat.copy()
    view_mat_without_translation[3][:3] = [0,0,0]

    # compute the inverse of the view (one without translation)- projection matrix
    inverseViewProjection_mat = pyrr.matrix44.inverse(pyrr.matrix44.multiply(view_mat_without_translation,projection_mat))

    glDepthFunc(GL_LEQUAL)    # Change depth function so depth test passes when values are equal to depth buffer's content
    glUseProgram(shaderProgram_skybox.shader)  # being explicit even though the line below will call this function
    shaderProgram_skybox["invViewProjectionMatrix"] = inverseViewProjection_mat
    glBindVertexArray(vao_quad)
    glDrawArrays(GL_TRIANGLES,
                 0,
                 quad_n_vertices)  # Draw the triangle
    glDepthFunc(GL_LESS)      # Set depth function back to default
    # *************************************************************


def render_obj():
    glUseProgram(shaderProgram_rayman.shader)

    shaderProgram_rayman["view_matrix"] = view_mat
    shaderProgram_rayman["projection_matrix"] = projection_mat
    shaderProgram_rayman["model_matrix"] = model_mat_obj
    shaderProgram_rayman["angle"] = float(y_angle)

    # Activate texture unit 0 and bind the texture to it
    # glActiveTexture(GL_TEXTURE0)
    # glBindTexture(GL_TEXTURE_2D, obj_tex_id)

    # Draw the triangle
    glBindVertexArray(vao_obj)      # Bind the VAO. That is, make it the active one.
    glDrawArrays(GL_TRIANGLES,
                 0,
                 n_vertices_obj)



'''
# Program starts here
'''


# Initialize pygame
pg.init()

FPS = 60
fpsClock = pg.time.Clock()
# Set up OpenGL context version
pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_STENCIL_SIZE, 8)

# Create a window for graphics using OpenGL

width = 1280
height = 720
display = (width, height)
screen = pg.display.set_mode(display)


draw = False

start_game = False
while not start_game:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            start_game = True 

    start_game = menu.draw_menu(screen)
    if start_game:
        draw = True

    pg.display.flip()
pg.quit()


pg.display.set_mode(display, pg.OPENGL | pg.DOUBLEBUF)

# Set the background color (clear color)
# glClearColor takes 4 arguments: red, green, blue, alpha. Each argument is a float between 0 and 1.
glClearColor(0.3, 0.4, 0.5, 1.0)
glEnable(GL_DEPTH_TEST)


# Write our shaders.
# Shader to generate the depth texture (shadown map) from light's point of view
shaderProgram_shadowMap = shaderLoaderV3.ShaderProgram("shaders/shadowMap/vert_shadowMap.glsl", "shaders/shadowMap/frag_shadowMap.glsl")
# optional: to render the depth texture onto a quad. Only used for debugging purposes
shaderProgram_visualizeTex = shaderLoaderV3.ShaderProgram("shaders/visualizeDepthTex/vert_tex.glsl", "shaders/visualizeDepthTex/frag_tex.glsl")
# Shader to render the scene with shadow from camera's point of view
shaderProgram_scene = shaderLoaderV3.ShaderProgram("shaders/scene/vert_scene.glsl", "shaders/scene/frag_scene.glsl")

shaderProgram_skybox = shaderLoaderV3.ShaderProgram("shaders/skybox/vert_skybox.glsl", "shaders/skybox/frag_skybox.glsl")

shaderProgram_rayman = shaderLoaderV3.ShaderProgram("shaders/mainRay/vert.glsl", "shaders/mainRay/frag.glsl")


'''
# **************************************************************************************************************
# Load our objects and configure their attributes
# **************************************************************************************************************
'''
obj = ObjLoader("objects/santasledge.obj")
#obj = ObjLoader("objects/raymanModel.obj")
vao_obj, vbo_obj, n_vertices_obj = upload_and_configure_attributes(obj, shaderProgram_rayman.shader)

obj_receiver = ObjLoader("objects/square.obj")
vao_receiver, vbo_receiver, n_vertices_receiver = upload_and_configure_attributes(obj_receiver, shaderProgram_scene.shader)

obj_tree = ObjLoader("objects/decoratedTree.obj")
vao_tree, vbo_tree, n_vertices_tree = upload_and_configure_attributes(obj_tree, shaderProgram_scene.shader)

obj_gift = ObjLoader("objects/GiftBox.obj")
vao_gift, vbo_gift, n_vertices_gift = upload_and_configure_attributes(obj_gift, shaderProgram_scene.shader)
# **************************************************************************************************************
# **************************************************************************************************************



'''
# **************************************************************************************************************
# Define camera attributes
# **************************************************************************************************************
'''
eye = (0,3,4)
target = (0,0,0)
up = (0,1,0)

fov = 63
aspect = width/height
near = 0.1
far = 20

lightPos = [1, 4, 1]
# **************************************************************************************************************
# **************************************************************************************************************



'''
# **************************************************************************************************************
# Configure model matrices
# **************************************************************************************************************
'''
# for object

# for receiver
floorcords = [0,-1,0]
rotation_mat = pyrr.matrix44.create_from_x_rotation(np.deg2rad(90))
translation_mat = pyrr.matrix44.create_from_translation(floorcords)
scaling_mat = pyrr.matrix44.create_from_scale([3, 3, 1])
model_mat_receiver = pyrr.matrix44.multiply(scaling_mat, rotation_mat)
model_mat_receiver = pyrr.matrix44.multiply(model_mat_receiver, translation_mat)

translation_tree = pyrr.matrix44.create_from_translation([1.2,-1.57,0.5])
scaling_tree = pyrr.matrix44.create_from_scale([1/5,1/5,1/5])
model_mat_tree = pyrr.matrix44.multiply(scaling_tree,translation_tree)

translation_gift = pyrr.matrix44.create_from_translation([-1.2,-0.77,-1])
scaling_gift = pyrr.matrix44.create_from_scale([1/850,1/850,1/850])
model_mat_gift = pyrr.matrix44.multiply(scaling_gift, translation_gift)
# **************************************************************************************************************
# **************************************************************************************************************

# translate_mat1 = pyrr.matrix44.create_from_translation(-center)


'''
# **************************************************************************************************************
# Framebuffer
# **************************************************************************************************************
'''
framebuffer_id, depthTex_id = create_framebuffer_with_depth_attachment(width, height)
shaderProgram_scene["depthTex"] = 0
shaderProgram_visualizeTex["depthTex"] = 0





'''
# **************************************************************************************************************
# Set up vertices, VAO, VBO, and vertex attributes for a quad that will be used to render the skybox
# **************************************************************************************************************
'''
# Define the vertices of the quad.
quad_vertices = (
            # Position
            -1, -1,
             1, -1,
             1,  1,
             1,  1,
            -1,  1,
            -1, -1
)
vertices = np.array(quad_vertices, dtype=np.float32)

size_position = 2       # x, y, z
stride = size_position * 4
offset_position = 0
quad_n_vertices = len(vertices) // size_position  # number of vertices

# Create VA0 and VBO
vao_quad = glGenVertexArrays(1)
glBindVertexArray(vao_quad)            # Bind the VAO. That is, make it the active one.
vbo_quad = glGenBuffers(1)                  # Generate one buffer and store its ID.
glBindBuffer(GL_ARRAY_BUFFER, vbo_quad)     # Bind the buffer. That is, make it the active one.
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)   # Upload the data to the GPU.


position_loc = 0
glBindAttribLocation(shaderProgram_skybox.shader, position_loc, "position")
glVertexAttribPointer(position_loc, size_position, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc)

'''
# **************************************************************************************************************
# Load the 2D texture and attach the sampler variable "tex" in the object shader to texture unit 0.
# **************************************************************************************************************
'''
obj_tex_id = load_2d_texture("objects/rayman/raymanModel.png")
# **************************************************************************************************************
# **************************************************************************************************************


'''
# **************************************************************************************************************
# Load the cubemap texture and attach it to texture unit 0 (GL_TEXTURE0). 
# **************************************************************************************************************
'''
cubemap_images = ["images/skybox2/right.jpg", "images/skybox2/left.jpg",
                  "images/skybox2/top.jpg", "images/skybox2/bottom.jpg",
                  "images/skybox2/front.jpg", "images/skybox2/back.jpg"]
cubemap_id = load_cubemap_texture(cubemap_images)


shaderProgram_skybox["cubeMapTex"] = 0   # Okay this might be confusing. Here 0 indicates texture unit 0. Note that "cubeMapTex" is a sampler variable in the fragment shader. It is not an integer.



# **************************************************************************************************************
# **************************************************************************************************************




'''
# **************************************************************************************************************
# other
# **************************************************************************************************************
'''

counter = 0
# Movement properties
tx, ty, tz = 0, 0, 0
move_speed = .5
jump_dist = 8.0

# Camera properties
camera_rotate_rate = 5
x_angle = 0
y_angle = 0


# **************************************************************************************************************
# **************************************************************************************************************




'''
# **************************************************************************************************************
# Setup gui
# **************************************************************************************************************
'''
gui = SimpleGUI("Skybox")

# Create a slider for the rotation angle around the Y axis
light_rotY_slider = gui.add_slider("light Y angle", -180, 180, 0, resolution=1)

# camera_rotY_slider = gui.add_slider("camera Y angle", -180, 180, 0, resolution=1)
# camera_rotX_slider = gui.add_slider("camera X angle", -90, 90, -32, resolution=1)
fov_slider = gui.add_slider("fov", 0, 120, fov, resolution=1)
gift_rotation = gui.add_slider("gift_rotation", 0,360, 0, resolution=1)

# character_slider = gui.add_slider("obj x-axis", -60,60, obj.center[1], resolution=1)
character_slider2 = gui.add_slider("obj y-axis", -70,30, -16, resolution=1)
# character_slider3 = gui.add_slider("obj z-axis", -60,60, obj.center[1], resolution=1)

#material_color_slider = gui.add_color_picker(label_text="Material Color", initial_color=(0.8, 0.8, 0.8))
render_type_radio = gui.add_radio_buttons(label_text="Render Type",
                                          options_dict={"Depth (Light's POV)": 0,"Scene with shadow": 1,},
                                          initial_option="Scene with shadow")

collision_checkbox= gui.add_checkbox("Collision", initial_state = True)
# **************************************************************************************************************
# **************************************************************************************************************



pg.init()

glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)   

# Timer
countdown = 10


TIMEREVENT = pg.USEREVENT + 1
pg.time.set_timer(TIMEREVENT, 1000)

draw1 = False

# Run a loop to keep the program running
while draw:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            draw = False
        # tempy = model_mat_obj[3][0]
        # tempx = model_mat_obj[3][1]
        # tempz = model_mat_obj[3][2]
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                # Move the object to the left
                ty += jump_dist
            print(pg.key.name(event.key))
        if event.type == pg.KEYUP:
            if event.key == pg.K_SPACE:
                # Move the object to the left
                ty -= jump_dist 
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                y_angle = (y_angle - 90) % 360
            elif event.button == 3:  # Right mouse button
                y_angle = (y_angle + 90) % 360
        if event.type == TIMEREVENT:
            # pg.display.set_caption(f"Countdown: {minutes:02}:{seconds:02}")
            countdown -= 1
            if countdown == 0:
                draw1 = True
                draw = False
                menu.drawText(580, 650, "Time\'s up!", (255, 0, 0))
                


    keys = pg.key.get_pressed()

    if (y_angle == 0):
        if keys[pg.K_a]:
            tx -= move_speed
        if keys[pg.K_d]:
            tx += move_speed
        if keys[pg.K_w]:
            ty += move_speed
        if keys[pg.K_s]:
            ty -= move_speed

    if (y_angle == 90):
        if keys[pg.K_a]:
            ty += move_speed
        if keys[pg.K_d]:
            ty -= move_speed
        if keys[pg.K_w]:
            tx += move_speed
        if keys[pg.K_s]:
            tx -= move_speed

    if (y_angle == 180):
        if keys[pg.K_a]:
            tx += move_speed
        if keys[pg.K_d]:
            tx -= move_speed
        if keys[pg.K_w]:
            ty -= move_speed
        if keys[pg.K_s]:
            ty += move_speed

    if (y_angle == 270):
        if keys[pg.K_a]:
            ty -= move_speed
        if keys[pg.K_d]:
            ty += move_speed
        if keys[pg.K_w]:
            tx -= move_speed
        if keys[pg.K_s]:
            tx += move_speed


    # Clear color buffer and depth buffer before drawing each frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Counter
    menu.drawText(1000, 650, "Gifts: "+str(counter), (225, 225, 0))

    seconds = countdown % 60
    minutes = int(countdown / 60) % 60
    # Timer
    menu.drawText(580, 650, f"{minutes:02}:{seconds:02}", (255, 0, 0))


    # view and projection matrices for camera's point of view
    cam_rotY_mat = pyrr.matrix44.create_from_y_rotation(np.deg2rad(y_angle))
    cam_rotX_mat = pyrr.matrix44.create_from_x_rotation(np.deg2rad(-20))
    cam_rot_mat = pyrr.matrix44.multiply(cam_rotX_mat, cam_rotY_mat)
    rotated_eye = pyrr.matrix44.apply_to_vector(cam_rot_mat, eye)

    view_mat = pyrr.matrix44.create_look_at(rotated_eye, target, up)
    #view_mat_rayman = pyrr.matrix44.create_look_at(eye, target, up)
    projection_mat = pyrr.matrix44.create_perspective_projection_matrix(fov_slider.get_value(), aspect, near,  far)

    # view and projection matrices for light's point of view
    light_rotY_mat = pyrr.matrix44.create_from_y_rotation(np.deg2rad(light_rotY_slider.get_value()))
    rotated_lightPos = pyrr.matrix44.apply_to_vector(light_rotY_mat, lightPos)

    light_view_mat = pyrr.matrix44.create_look_at(rotated_lightPos, target, up)
    light_projection_mat = pyrr.matrix44.create_perspective_projection_matrix(fov_slider.get_value(), aspect, near, far)

    obj.center[0] = tx
    obj.center[1] = ty
    obj.center[2] = character_slider2.get_value()


    translation_mat = pyrr.matrix44.create_from_translation([obj.center[0],obj.center[1],obj.center[2]])
    scaling_mat = pyrr.matrix44.create_from_scale([1 / 12, 1 / 12, 1 / 12])
    model_mat_obj = pyrr.matrix44.multiply(translation_mat, scaling_mat)
    model_mat_obj = pyrr.matrix44.multiply(model_mat_obj, rotation_mat)


    gift_rotation2 = gift_rotation.get_value()

    if collision_checkbox.get_value() == True:
        # collision with the ground
        collision.check_collision(model_mat_obj, model_mat_receiver, translation_mat, scaling_mat)

        # collision with the walls
        # collision.check_collision2(model_mat_obj,model_mat_receiver,translation_mat,scaling_mat)

        # collision with the tree
        collision.check_collison3(model_mat_obj,model_mat_tree,ty,tx)

        # collision with the gift
        counter = collision.check_collision4(model_mat_obj,model_mat_gift, counter)

    # render the shadow map. The object and the receiver will be rendered from the light's point of view
    # and the depth of each fragment will be stored in a texture (depthTex_id) which will be used in the next part
    render_shadow_map()

    render_skybox()

    render_obj()

    if(int(render_type_radio.get_value()) == 0):
        render_tex()        # optional: render the depth texture onto a quad for debugging purposes
    else:
        render_scene()      # render the scene with shadow. You need to implement this function

    # Refresh the display to show what's been drawn
    pg.display.flip()
    fpsClock.tick(FPS)

pg.init()

pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_STENCIL_SIZE, 8)

width = 1280
height = 720
display = (width, height)
screen = pg.display.set_mode(display)

asd = False
while draw1:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            draw1 = True 

    asd = menu.draw_result(screen, str(counter))

    pg.display.flip()
pg.quit()


# Cleanup
glDeleteVertexArrays(2, [vao_obj, vao_receiver])
glDeleteBuffers(2, [vbo_obj, vao_receiver])

glDeleteProgram(shaderProgram_scene.shader)
glDeleteProgram(shaderProgram_shadowMap.shader)

pg.quit()   # Close the graphics window
quit()      # Exit the program