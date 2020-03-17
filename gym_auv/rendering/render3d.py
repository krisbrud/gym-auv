from __future__ import division

import sys
import math
import random
import time
import numpy as np
import logging
import pywavefront
from pywavefront import visualization

pywavefront.configure_logging(
    logging.FATAL,
    formatter=logging.Formatter('%(name)s-%(levelname)s: %(message)s')
)

from collections import deque
from pyglet import image
from pyglet.gl import *
from pyglet.graphics import TextureGroup
from pyglet.window import key, mouse
import gym_auv.utils.geomutils as geom

# Size of sectors used to ease block loading.
MAX_RENDER_DISTANCE = 1000.0
PAD = 50
SECTOR_SIZE = 25
FOG_DISTANCE = 1000
TICKS_PER_SEC = 1 
CAMERA_ROTATION_SPEED = 0.02
SKY_COLOR = (109/255, 173/255, 255/255, 1)
ENABLE_LIGHT = True

WINDOW_W = 1920 # 720
WINDOW_H = 1080 # 600

counter_3d = 0

BOAT_MODEL_PATH = './resources/boat.obj'

class Element:
    BLOCK = 1
    PLANE = 2

def cube_vertices(x, z, y_00, y_01, y_10, y_11, n):
    """ Return the vertices of the cube at position x, y, z with size 2*n.

    """

    return [
        x-n,y_00,z-n, x-n,y_01,z+n, x+n,y_11,z+n, x+n,y_10,z-n,  # top
        x-n,0,z-n, x+n,0,z-n, x+n,0,z+n, x-n,0,z+n,  # bottom
        x-n,0,z-n, x-n,0,z+n, x-n,y_01,z+n, x-n,y_00,z-n,  # left
        x+n,0,z+n, x+n,0,z-n, x+n,y_10,z-n, x+n,y_11,z+n,  # right
        x-n,0,z+n, x+n,0,z+n, x+n,y_11,z+n, x-n,y_01,z+n,  # front
        x+n,0,z-n, x-n,0,z-n, x-n,y_00,z-n, x+n,y_10,z-n,  # back
    ]

def plane_vertices(x, z, y_00, y_01, y_10, y_11, n):return [
        x-n,y_00,z-n, x-n,y_01,z+n, x+n,y_11,z+n, x+n,y_10,z-n
    ]


def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square.

    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m


def tex_coords(top, bottom, side):
    """ Return a list of the texture squares for the top, bottom and side.

    """
    top = tex_coord(*top)
    bottom = tex_coord(*bottom)
    side = tex_coord(*side)
    result = []
    result.extend(top)
    result.extend(bottom)
    result.extend(side * 4)
    return result

def get_neighbours(x, y, matrix, include_diag=False):
    ans = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            if not include_diag:
                if abs(dx) == 1 and abs(dy) == 1:
                    continue
            
            p = (x + dx, y + dy)

            if p[0] >= 0 and p[0] < matrix.shape[0] and p[1] >= 0 and p[1] < matrix.shape[1]:
                ans.add(p)

    return ans

TEXTURE_PATH = './resources/textures.png'

DIRT_GRASS = tex_coords((1, 0), (0, 1), (0, 0))
DIRT = tex_coords((0, 1), (0, 1), (0, 1))
GRASS = tex_coords((1, 0), (1, 0), (1, 0))
SAND = tex_coords((1, 1), (1, 1), (1, 1))
BRICK = tex_coords((2, 0), (2, 0), (2, 0))
STONE = tex_coords((2, 1), (2, 1), (2, 1))
ICE = tex_coords((3, 0), (3, 0), (3, 0))
WATER = tex_coords((0.5, 2.5), (0.5, 2.5), (0.5, 2.5))

FACES = [
    ( 0, 1, 0),
    ( 0,-1, 0),
    (-1, 0, 0),
    ( 1, 0, 0),
    ( 0, 0, 1),
    ( 0, 0,-1),
]

def normalize(position):
    """ Accepts `position` of arbitrary precision and returns the block
    containing that position.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    block_position : tuple of ints of len 3

    """
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return (x, y, z)


def sectorize(position):
    """ Returns a tuple representing the sector for the given `position`.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    sector : tuple of len 3

    """
    x, y, z = normalize(position)
    x, y, z = x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE
    return (x, 0, z)


class Viewer3D(object):
    def __init__(self, width, height, autocamera=False):
        self.autocamera = autocamera

        self.overlay_batch = pyglet.graphics.Batch()
        self.group = TextureGroup(image.load(TEXTURE_PATH).get_texture())
        self.path = {}

        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height)
        #self.window.maximize()

        self.camera_height = 13
        self.camera_distance = 45
        self.camera_angle = 0

        if self.autocamera:
            self.camera_height_goal = self.camera_height
            self.camera_distance_goal = self.camera_distance
            self.camera_angle_goal = self.camera_angle

        self.rotation = (0, 0)
        self.boat_models = {}
        
        self.main_batch = pyglet.graphics.Batch()
        self.world = {}
        self.terrain_shape = {}
        self.element_type = {} 
        self.shown = {}
        self._shown = {}
        self.position = (0, self.camera_height, 0)
        self.queue = deque()

        self.reset_world()

        self.label = pyglet.text.Label('', font_size=18,
            x=210, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def reset_world(self):
        #self.queue = deque()
        self.sectors = {}
        self.sector = None

    def create_path(self, path):
        for p in path.points:
            self.add_element((p[1], 0, p[0]), 2*[1.0, 1.0, 1.0, 1.0], 4*[0.1,], Element.PLANE)

    def create_world(self, terrain, xlow, ylow, xhigh, yhigh):
        self.reset_world()
        N = (xhigh-xlow)*(yhigh-ylow)
        for x in range(xlow, xhigh):
            for y in range(ylow, yhigh):

                z = terrain[x][y]
                posarr = (y, z, x)
                if posarr in self.world:
                    continue

                if x == xlow or x == (xhigh-1)  or y == ylow or y == (yhigh-1):
                    h_00, h_01, h_10, h_11 = (z, z, z, z)
                
                else:
                    h_00 = np.mean([z, terrain[x-1][y], terrain[x-1][y-1], terrain[x][y-1]])
                    h_01 = np.mean([z, terrain[x-1][y], terrain[x-1][y+1], terrain[x][y+1]])
                    h_10 = np.mean([z, terrain[x+1][y], terrain[x+1][y-1], terrain[x][y-1]])
                    h_11 = np.mean([z, terrain[x+1][y], terrain[x+1][y+1], terrain[x][y+1]])


                if z<=0: # h_00 == 0 and h_10 == 0 and h_01 == 0 and h_11 == 0:
                    t = WATER
                else:
                    neighbours = get_neighbours(x, y, matrix=terrain)
                    values = np.array([terrain[p[0]][p[1]] for p in neighbours])

                    if np.std(values) >= 0.4:
                        t = DIRT

                    else:
                        t = ICE

                self.add_element(posarr, t, (h_00, h_10, h_01, h_11), Element.BLOCK)

                i = (x-xlow)*(yhigh-ylow) + (y-ylow)

                # if i % 100 == 0:
                #     sys.stdout.write('Creating {}x{} world ({:.2%})\r'.format((xhigh-xlow), (yhigh-ylow), i/N))
                #     sys.stdout.flush()
                
    def close(self):
        self.window.close()

    def update(self):

        """ This method is scheduled to be called repeatedly by the pyglet
        clock.
        """
        self.process_queue()
        sector = sectorize(self.position)
        if sector != self.sector:
            self.change_sectors(self.sector, sector)
            if self.sector is None:
                self.process_entire_queue()
            self.sector = sector

    def exposed(self, position):
        """ Returns False is given `position` is surrounded on all 6 sides by
        blocks, True otherwise.

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            if (x + dx, y + dy, z + dz) not in self.world:
                return True
        return False

    def add_element(self, position, texture, shape, element_type):
        """ Add a block with the given `texture` and `position` to the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to add.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.

        """
        self.terrain_shape[position] = shape
        self.world[position] = texture
        self.element_type[position] = element_type
        self.sectors.setdefault(sectorize(position), []).append(position)

    def remove_element(self, position):
        """ Remove the block at the given `position`.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to remove.

        """
        del self.world[position]
        del self.terrain_shape[position]
        del self.element_type[position]
        self.sectors[sectorize(position)].remove(position)

    def show_element(self, position):
        """ Show the block at the given `position`. This method assumes the
        block has already been added with add_element()

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to show.

        """
        texture = self.world[position]
        shape = self.terrain_shape[position]
        element_type = self.element_type[position]
        self.shown[position] = texture
        self._enqueue(self._show_element, position, texture, shape, element_type)

    def _show_element(self, position, texture, shape, element_type):
        """ Private implementation of the `show_element()` method.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to show.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.

        """
        x, y, z = position
        texture_data = list(texture)

        if element_type == Element.BLOCK:
            vertex_data = cube_vertices(x, z, shape[0], shape[1], shape[2], shape[3], 0.5)
            self._shown[position] = self.main_batch.add(24, GL_QUADS, self.group,
            ('v3f/static', vertex_data),
            ('t2f/static', texture_data))
        elif element_type == Element.PLANE:
            vertex_data = plane_vertices(x, z, shape[0], shape[1], shape[2], shape[3], 0.1)
            self._shown[position] = self.overlay_batch.add(4, GL_QUADS, self.group, ('v3f/static', vertex_data),  ('t2f/static', texture_data))

    def hide_element(self, position):
        """ Hide the block at the given `position`. Hiding does not remove the
        block from the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to hide.

        """
        self.shown.pop(position)
        self._enqueue(self._hide_element, position)

    def _hide_element(self, position):
        """ Private implementation of the 'hide_element()` method.

        """
        self._shown.pop(position).delete()

    def show_sector(self, sector):
        """ Ensure all blocks in the given sector that should be shown are
        drawn to the canvas.

        """
        for position in self.sectors.get(sector, []):
            if position not in self.shown and self.exposed(position):
                self.show_element(position)

    def hide_sector(self, sector):
        """ Ensure all blocks in the given sector that should be hidden are
        removed from the canvas.

        """
        for position in self.sectors.get(sector, []):
            if position in self.shown:
                self.hide_element(position)

    def change_sectors(self, before, after):
        """ Move from sector `before` to sector `after`. A sector is a
        contiguous x, y sub-region of world. Sectors are used to speed up
        world rendering.

        """
        #print('Moving from sector ', before, 'to', after)
        before_set = set()
        after_set = set()
        for dx in range(-PAD, PAD + 1):
            for dy in [0]:  # range(-pad, pad + 1):
                for dz in range(-PAD, PAD + 1):
                    if dx ** 2 + dy ** 2 + dz ** 2 > (PAD + 1) ** 2:
                        continue
                    if before:
                        x, y, z = before
                        before_set.add((x + dx, y + dy, z + dz))
                    if after:
                        x, y, z = after
                        after_set.add((x + dx, y + dy, z + dz))
        show = after_set - before_set
        hide = before_set - after_set
        for sector in show:
            self.show_sector(sector)
        for sector in hide:
            self.hide_sector(sector)

    def _enqueue(self, func, *args):
        """ Add `func` to the internal queue.

        """
        self.queue.append((func, args))

    def _dequeue(self):
        """ Pop the top function from the internal queue and call it.

        """
        func, args = self.queue.popleft()
        func(*args)

    def process_queue(self):
        """ Process the entire queue while taking periodic breaks. This allows
        the game loop to run smoothly. The queue contains calls to
        _show_element() and _hide_element() so this method should be called if
        add_element() or remove_element() was called with immediate=False

        """
        start = time.clock()
        while self.queue and time.clock() - start < 1.0 / TICKS_PER_SEC:
            self._dequeue()

    def process_entire_queue(self):
        """ Process the entire queue with no breaks.

        """
        while self.queue:
            self._dequeue()

    def __del__(self):
        self.close()

    def draw_label(self, env):
        global counter_3d
        """ Draw the label in the top left of the screen.

        """
        # x, y, z = self.position
        # xrot, yrot = self.rotation
        self.label.text = 'Lg. λ: {:.2f}'.format(np.log10(env.config["reward_lambda"]))
        # self.label.text = '(%.2f, %.2f, %.2f) (%.2f, %.2f) %d / %d : %d'  % (
        #     z, x, y, xrot, yrot,
        #     len(self._shown), len(self.world), counter_3d)
        self.label.draw()

    def set_2d(self):
        """ Configure OpenGL to draw in 2d.

        """
        width, height = self.window.get_size()
        glDisable(GL_DEPTH_TEST)
        #viewport = self.window.get_viewport_size()
        #glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, max(1, width), 0, max(1, height), -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d(self):
        global counter_3d
        """ Configure OpenGL to draw in 3d.

        """
        width, height = self.window.get_size()
        glEnable(GL_DEPTH_TEST)

        # glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(0,0,0,1))
        # glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(0,0,0,1))
        # glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat*4)(1,1,1,1))

        # [...]


        #glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        #viewport = self.window.get_viewport_size()
        #glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if ENABLE_LIGHT:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            lightpos = (-1.0,1.0,1.0,0)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(0.35, 0.35, 0.5, 1.0))
            glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.35, 0.35, 0.5, 1.0))
            #glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, GLfloat(0.1))
            #glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, GLfloat(0.1))
            # glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, GLfloat(0.5))
            # glLightfv(GL_LIGHT0, GL_SPOT_CUTOFF, GLfloat(180.0))

            # glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, (GLfloat*3)(np.sin(0.01*counter_3d), 0, np.cos(0.01*counter_3d)))
            # #glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(*lightpos))
            # glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(1.0,1.0,1.0,1))
            # glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat*4)(1.0,1.0,1.0,1))
            # glLightfv(GL_LIGHT0, GL_SPOT_EXPONENT, GLfloat(0.01)) #*counter_3d))
            # glLightfv(GL_LIGHT0, GL_CONSTANT_ATTENUATION, GLfloat(0.001*counter_3d))
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
            glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(0.01))

        #glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, (GLfloat*1)(1))
        # glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat*4)(1,1,1,1))

        gluPerspective(65.0, width / float(height), 0.1, MAX_RENDER_DISTANCE)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        x, y = self.rotation
        glRotatef(x, 0, 1, 0)
        

        glRotatef(-y, math.cos(math.radians(x)), 0, math.sin(math.radians(x)))

        x, y, z = self.position
        glTranslatef(-x, -y, -z)
        
        if ENABLE_LIGHT:
            glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(*lightpos))

        counter_3d += 1
        
        


def render_env(env, mode, dt):
    x, y, z = (env.vessel.y, env.viewer3d.camera_height, env.vessel.x)

    if env.viewer3d.autocamera:
        d_height = env.viewer3d.camera_height_goal - env.viewer3d.camera_height
        d_distance = env.viewer3d.camera_distance_goal - env.viewer3d.camera_distance
        #d_angle = env.viewer3d.camera_angle_goal - env.viewer3d.camera_angle

        if abs(d_height) < 1 and abs(d_distance) < 1: # and abs(d_angle) < 1:
            env.viewer3d.camera_height_goal = np.random.gamma(shape=1.0, scale=25.0)
            env.viewer3d.camera_distance_goal = max(15, np.random.gamma(shape=1.0, scale=50.0))
            #env.viewer3d.camera_angle_goal = (-180 + 360*np.random.random()) if np.random.random() > 0.5 else env.vessel.heading*90

        env.viewer3d.camera_height += CAMERA_ROTATION_SPEED*d_height
        env.viewer3d.camera_distance += CAMERA_ROTATION_SPEED*d_distance
        #env.viewer3d.camera_angle += CAMERA_ROTATION_SPEED*180/np.pi*geom.princip(d_angle*np.pi/180)

    if env.viewer3d.camera_angle is None:
        env.viewer3d.camera_angle = -env.vessel.heading*180/np.pi + 180
    else:
        env.viewer3d.camera_angle += CAMERA_ROTATION_SPEED*180/np.pi*geom.princip(-env.vessel.heading + np.pi - env.viewer3d.camera_angle*np.pi/180)

    camera_direction = (-env.viewer3d.camera_angle + 180)*np.pi/180
    camera_x = x-np.sin(camera_direction)*env.viewer3d.camera_distance
    camera_z = z-np.cos(camera_direction)*env.viewer3d.camera_distance
    try:
        camera_terrain_height = env.all_terrain[int(camera_z)][int(camera_x)]
    except IndexError:
        camera_terrain_height = 0
    y = max(camera_terrain_height+1, y)

    #print(env.viewer3d.camera_angle, 180/np.pi*env.vessel.heading, env.vessel.x, env.vessel.y, camera_x, camera_z)

    env.viewer3d.position = (camera_x, y, camera_z) 
    env.viewer3d.rotation = (env.viewer3d.camera_angle, -180/np.pi*np.arctan2(y, env.viewer3d.camera_distance))
    env.viewer3d.update()
    env.viewer3d.window.switch_to()
    env.viewer3d.window.dispatch_events()
    env.viewer3d.window.clear()
    
    env.viewer3d.set_3d()
    glColor3d(1, 1, 1)
    
    # render other vessels
    for obsvessel in env.vessel_obstacles:
        glTranslatef(obsvessel.position[1], -1, obsvessel.position[0])
        glRotatef(-90, 1, 0, 0)
        glRotatef(90 + obsvessel.heading*180/np.pi, 0, 0, 1)
        if obsvessel.width not in env.viewer3d.boat_models:
            save_boatmodel(obsvessel.width, env)
        visualization.draw(env.viewer3d.boat_models[obsvessel.width])
        glRotatef(- 90 - obsvessel.heading*180/np.pi, 0, 0, 1)
        glRotatef(90, 1, 0, 0)
        glTranslatef(-obsvessel.position[1], 1, -obsvessel.position[0])

    # render agent vessel
    glTranslatef(x, -1, z)
    glRotatef(-90, 1, 0, 0)
    glRotatef(90 + env.vessel.heading*180/np.pi, 0, 0, 1)
    visualization.draw(env.viewer3d.boat_models[env.vessel.width])
    glRotatef(-90 - env.vessel.heading*180/np.pi, 0, 0, 1)
    glRotatef(90, 1, 0, 0)
    glTranslatef(-x, 1, -z)

    env.viewer3d.main_batch.draw()
    
    if ENABLE_LIGHT:
        glDisable(GL_LIGHTING)
    env.viewer3d.overlay_batch.draw()
    if ENABLE_LIGHT:
        glEnable(GL_LIGHTING)
    env.viewer3d.set_2d()
    env.viewer3d.draw_label(env)

    x_shift = 0#-170
    glViewport(x_shift, 0, WINDOW_W, WINDOW_H)

    arr = None
    if mode == 'rgb_array':
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(WINDOW_H, WINDOW_W, 4)
        arr = arr[::-1, :, 0:3]

    env.viewer3d.window.flip()

    return arr


def setup_fog():
    """ Configure the OpenGL fog properties.

    """
    # Enable fog. Fog "blends a fog color with each rasterized pixel fragment's
    # post-texturing color."
    glEnable(GL_FOG)
    # Set the fog color.
    glFogfv(GL_FOG_COLOR, (GLfloat * 4)(*SKY_COLOR))
    # Say we have no preference between rendering speed and quality.
    glHint(GL_FOG_HINT, GL_DONT_CARE)
    # Specify the equation used to compute the blending factor.
    glFogi(GL_FOG_MODE, GL_LINEAR)
    # How close and far away fog starts and ends. The closer the start and end,
    # the denser the fog in the fog range.
    glFogf(GL_FOG_START, 20.0)
    glFogf(GL_FOG_END, FOG_DISTANCE)

def setup():
    """ Basic OpenGL configuration.

    """
    # Set the color of "clear", i.e. the sky, in rgba.
    glClearColor(*SKY_COLOR)
    # Enable culling (not rendering) of back-facing facets -- facets that aren't
    # visible to you.
    glEnable(GL_CULL_FACE)
    # Set the texture minification/magnification function to GL_NEAREST (nearest
    # in Manhattan distance) to the specified texture coordinates. GL_NEAREST
    # "is generally faster than GL_LINEAR, but it can produce textured images
    # with sharper edges because the transition between texture elements is not
    # as smooth."
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    setup_fog()

def init_env_viewer(env, autocamera=False):
    env.viewer3d = Viewer3D(WINDOW_W, WINDOW_H, autocamera=autocamera)
    setup()

def save_boatmodel(width, env):
    env.viewer3d.boat_models[width] = pywavefront.Wavefront(BOAT_MODEL_PATH)
    vertices = []
    npvertices = np.array(env.viewer3d.boat_models[width].vertices)
    MODEL_BOAT_LENGTH = npvertices[:, 0].max() - npvertices[:, 0].min()
    boat_scale = 2*width/MODEL_BOAT_LENGTH
    for v in env.viewer3d.boat_models[width].vertices:
        w = tuple((x*boat_scale for x in v))
        vertices.append(w)
    env.viewer3d.boat_models[width].vertices = vertices
    for name, material in env.viewer3d.boat_models[width].materials.items():
        material.vertices = tuple((x*boat_scale for x in material.vertices))
    return boat_scale
    
def init_boat(env):
    if env.vessel.width not in env.viewer3d.boat_models:
        boat_scale = save_boatmodel(env.vessel.width, env)
        #print('Initialized 3D vessel model, scale factor is {:.4f}'.format(boat_scale))