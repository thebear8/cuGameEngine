# cuGameEngine
This is a very small toy game engine I created to play around with cuda and computer graphics.

It's not at all complete, but has a few nice features. It's somewhat similar to OpenGL, but all functionality is implemented from scratch in cuda.

Right now, there's only very basic 2d functionality like surfaces, blitting, alpha-blending, loading images into surfaces and so on.
It does have a very basic render pipeline and support for shaders, which are written in cuda aswell.

It also has a decent sdf(signed distance field) text rendering system with support for custom fonts. It supports font smoothing/anti-aliasing, tabs(L'\t'), newlines(L'\n') and text wrapping, but only on character level, not word level.

One of the goals of this project was to have a lot of control over rendering, so the render pipeline is designed to be extremly basic and customizable.
Another goal of this project was to keep the code as simple, compact and easy to understand as possible.

It also doesn't have a concept of 3d, but it takes care of things like 2d surfaces and supporting functionality, keyboard and mouse input and presenting the rendered data in a window.

Right now, it only supports windows, but it should be possible to port it to linux or mac.
It obviously also needs a cuda capable GPU.

Feel free to use this code to experiment and try out stuff!
If you'd like to contribute, feel free to do so by sending me a pull request! I'll try to look into it as quickly as possible, but I can't promise anything.
