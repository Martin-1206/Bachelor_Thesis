import os

import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

xml_path = (
    "xml/digit-v3.xml"  # xml file (assumes this is in the same folder as this file)
)
simend = 20  # simulation time
print_camera_config = 0  # set to 1 to print camera config
# this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


def init_controller(model, data):
    # initialize the controller here. This function is called once, in the beginning
    pass


def controller(model, data):
    # put the controller here. This function is called inside the simulation.
    pass


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = (
        glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    )
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    # update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# initialize the controller here. This function is called once, in the beginning
cam.azimuth = 89.83044433593757
cam.elevation = -89.0
cam.distance = 5.04038754800176
cam.lookat = np.array([0.0, 0.0, 0.0])

# initialize the controller
init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)

N = 5000
q0_start = 0
q0_end = 1.57
q1_start = 0
q1_end = -2 * 3.14
q0 = np.linspace(q0_start, q0_end, N)
q1 = np.linspace(q1_start, q1_end, N)


positions = [
    0.3779345350539291,
    0.17445200000000005,
    0.7564919999999999,
    -1.24868,
    0.624161,
    -0.705946,
    -0.24899487673343607,
    -0.21893699999999996,
    -0.7145469999999999,
    1.2520899999999997,
    -0.6458750000000001,
    0.7005230000000001,
    -0.5224399999999999,
    2.29168,
    0.23309600000000003,
    -1.2816000000000003,
    0.6105479999999999,
    -2.022225718412943,
    -0.5509680000000001,
    0.7432860000000001,
    -8.79066086495858e-05,
    1.2412516971681757,
    -0.7617899999999999,
    -0.22482,
    0.004695567281833944,
    -0.0027719805971874395,
    -1.2396182874205357,
    0.774997,
    0.15297599999999997,
    -0.004757710828356771,
]
listOfJoints = [
    "left-hip-roll",
    "left-hip-yaw",
    "left-hip-pitch",
    "left-knee",
    "left-toe-A",
    "left-toe-B",
    "right-hip-roll",
    "right-hip-yaw",
    "right-hip-pitch",
    "right-knee",
    "right-toe-A",
    "right-toe-B",
    "left-shoulder-roll",
    "left-shoulder-pitch",
    "left-shoulder-yaw",
    "left-elbow",
    "right-shoulder-roll",
    "right-shoulder-pitch",
    "right-shoulder-yaw",
    "right-elbow",
    "left-shin",
    "left-tarsus",
    "left-toe-pitch",
    "left-toe-roll",
    "left-heel-spring",
    "right-shin",
    "right-tarsus",
    "right-toe-pitch",
    "right-toe-roll",
    "right-heel-spring",
]


qpos_ids = [model.jnt_qposadr[mj.mj_name2id(model, 3, joint)] for joint in listOfJoints]

dof_id = [mj.mj_name2id(model, 4, joint) for joint in listOfJoints]

print(qpos_ids)
# print(model.jnt_qposadr)
# for i in range(len(qpos_ids)):
#     model.qpos0[qpos_ids[i]] = positions[i]

# mj.mj_resetData(model, data)
# # exit()
# exit()

print(data.qpos)
print(len(data.qpos))
exit()

# initialize
data.qpos[0] = q0_start
data.qpos[1] = q1_start
i = 0
time = 0
dt = 0.0005


data.qpos[0] = 0
data.qpos[1] = 0
data.qpos[2] = 0.5

data.qpos[3] = -0.18946987384437597
# base quaternion w
data.qpos[4] = 0.5963002313174114
# base quaternion x
data.qpos[5] = 0.149640718798151
# base quaternion y
data.qpos[6] = 0.7655938975346687
# base quaternion z

mj.mj_forward(model, data)

while not glfw.window_should_close(window):
    time_prev = time

    while time - time_prev < 1.0 / 60.0:
        if data.time < 5:
            for m in range(len(qpos_ids)):
                data.qpos[qpos_ids[m]] = data.qpos[qpos_ids[m]] + np.clip(
                    data.time / 100, 0, 0.1
                ) * (positions[m] - data.qpos[qpos_ids[m]])
                model.dof_damping[dof_id[m]] = 1000

            # mj.mj_forward(model, data)
            for _ in range(5):
                mj.mj_step(model, data)
        else:
            data.qpos = [
                0,
                0,
                3.12123452e-01,
                7.94165858e-01,
                -3.18998841e-02,
                -6.05619803e-01,
                -3.88283705e-02,
                3.79563903e-01,
                1.71615852e-01,
                7.14872089e-01,
                5.40215447e-01,
                -6.00078506e-01,
                4.55130028e-01,
                -3.75406065e-01,
                -1.26728523e00,
                4.96896521e-03,
                1.24981243e00,
                4.59765447e-03,
                6.17704315e-01,
                8.92144230e-01,
                3.25845862e-01,
                -9.23525971e-02,
                -2.98955089e-01,
                -6.90285825e-01,
                6.44207045e-01,
                -6.76990275e-01,
                -2.70788182e-01,
                2.30987468e-01,
                -7.45268852e-01,
                -1.91818483e-01,
                -5.21987323e-01,
                2.29274893e00,
                2.33119136e-01,
                -1.28149154e00,
                -2.48861073e-01,
                -2.13349805e-01,
                -6.94524594e-01,
                1.85939986e-01,
                7.91741411e-01,
                5.71547519e-01,
                1.09111381e-01,
                1.24542582e00,
                8.57552216e-04,
                -1.24980086e00,
                1.97993153e-03,
                -6.36081142e-01,
                -8.91427863e-01,
                -3.17713178e-01,
                -1.22967698e-01,
                -2.98820424e-01,
                7.05082920e-01,
                7.15762540e-01,
                -5.97458803e-01,
                2.14972102e-01,
                -2.90712848e-01,
                7.67814943e-01,
                1.92393509e-01,
                6.05221296e-01,
                -1.99455030e00,
                -5.50476356e-01,
                7.43274428e-01,
            ]

            mj.mj_step(model, data)

        time += dt
        # for i in range(0, 4):
        #     mj.mj_step(model, data)

    i += 1
    print(data.qpos)

    if i >= N:
        break
    # if (data.time>=simend):
    #     break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # print camera configuration (help to initialize the view)
    if print_camera_config == 1:
        print(
            "cam.azimuth =",
            cam.azimuth,
            ";",
            "cam.elevation =",
            cam.elevation,
            ";",
            "cam.distance = ",
            cam.distance,
        )
        print(
            "cam.lookat =np.array([",
            cam.lookat[0],
            ",",
            cam.lookat[1],
            ",",
            cam.lookat[2],
            "])",
        )

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()


glfw.terminate()
