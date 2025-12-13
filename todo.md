* color maps
* normalize normals in vertex shader rather than fragment shader
* half precision positions, normals, intensities?
* stop putting height in texture
* align vertex order with shadow direction to reduce geometry pressure
* persistently mapped buffers - cuda and cpu
* synchronizing the two windows - threads - remove expensive calls to glfwMakeContextCurrent OR single window, 2 viewports?
* minimize driver overhead
    - bindless textures
    - direct state access
    - remove unnecessary gl calls
    - 3d window does not need texture switching or shadow mapping when running solo
* ambient occlusion