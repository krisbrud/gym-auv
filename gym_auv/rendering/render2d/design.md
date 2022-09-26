# Renderer
The renderer has been rewritten as a part of this project. While `EivMeyer/gym-auv` uses `pyglet` as the renderer backend, this project has opted for using `pygame`, which is also used for rendering in `OpenAI/gym`.

## Design

### `Renderer`
The renderer is the main class. The environment has a renderer, and the relevant state to be rendered is passed at each `.render()`-call

### `Geometry`
The geometry class describes low level geometry functions. 

### `Factories`
Factories makes geometry objects from the state of the environment. Factories may use the position, rotation and scale of the vessel to express the resulting geometries in body coordinates.
