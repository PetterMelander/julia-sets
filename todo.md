* color maps
* normalize normals in vertex shader rather than fragment shader
* half precision positions, normals, intensities?
* persistently mapped buffers - cuda and cpu
* synchronizing the two windows - threads - remove expensive calls to glfwMakeContextCurrent OR single window, 2 viewports?
* minimize driver overhead
    - bindless textures
    - direct state access
    - remove unnecessary gl calls
    - 3d window does not need texture switching or shadow mapping when running solo
* ambient occlusion