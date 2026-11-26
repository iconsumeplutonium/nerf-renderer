## The short version

Fire a ray from the camera through a pixel. Pick a bunch of points along it. For each point, look up learned features from the hash grid, run them through two small neural networks to get a density and a color, and then blend all the points along the ray into one final color. That color is the pixel.

## Step 1: pick sample points along the ray

For each ray, choose a number of evenly spread points on the ray between a near distance (2) and a far distance (6). The script uses 192 of them.

During training, each sample point is chosen randomly within each interval. For repeatable renders, always choose the middle of the interval. 

## Step 2: skip points outside the bounding box

The scene lives inside a cube of size 3x3x3 centered on the origin. Divide the point by 3, to make the cube -0.5 to 0.5 on every axis. If any coordinate is outside that range, those get density 0, color 0. For points inside, add 0.5 so the coordinates run 0 to 1, which is what the hash grid expects.

## Step 3: look up features from the hash grid

* For each sample point, look up its cell position in the grid
* Find the corners of the cell
* for each corner, look up its 2 learned numbers in the hash table
* interpolate between all 8 to get 2 learned numbers for our sample point
* repeat this for all grid resolutions 

After all 16 levels, the point has 16 x 2 = 32 numbers. This is the hash encoding of the point's position.

## Step 4: density network

The 32 numbers are passed into the density MLP. 32-in, ReLU, then 16 out.

The first of the 16 outputs is the log of the density. Take `exp` of it to get the actual density, which is always positive. 

## Step 5: color network

Color can depend on the viewing angle (highlights, reflections), so this is where the ray's direction comes in.

* Encode the ray direction with positional encoding at different frequencies, which produces 27 numbers.
* Combine the 16 density MLP outputs and the 27 direction numbers together, which gives 43 numbers
* Pass that vector into the color MLP. The result is an RGB triple between 0 and 1, represeting predicted color

The camera origin is never an input to either MLP. It is only used to place the sample points in step 1. After that the model knows a 3D position and a viewing direction, and nothing about where the camera is.

## Step 6: combine the samples into one pixel

Now every sample on the ray has a color and a density. The sample points outside the box scene were at density 0 and color 0, while the ones inside the box got their predicted density and color from the MLP. The ray's color is built front to back.

* Opacity for each sample is calculated as `alpha = 1 - exp(-density * distance between samples)`.
* The transmittance for each sample is calculated by accumulating `(1 - alpha)` repeatedly. So the transmittance of the first point is 1, the second point is 1 * (1-alpha_2), then the third point is 1 * (1-alpha_2) * (1-alpha_3), etc. 
* The weight for each sample is calculated as `transmittence * alpha`
* The color for the ray is calculated by adding up all `weight * predictedColor`
* Then we need to fill in the background, because the background is white not black. So you add `(1 - sum(weight for each sample))` to add that white back in

## Same thing as a loop

```
color = (0, 0, 0)
T = 1
for each sample, near to far:
    point = origin + t * direction
    if point is outside the box: continue
    features = hash grid lookup, 16 levels x 2 numbers
    h = density_network(features)            (16 numbers)
    density = exp(h[0])
    rgb = color_network(h + encoded direction)
    alpha = 1 - exp(-density * delta)
    color += T * alpha * rgb
    T *= (1 - alpha)
    if T is tiny: stop early
pixel = color + T          (T is the unblocked fraction, which shows white)
```


things to do:

* send model weights and biases into the gpu
* send hash table to the gpu
* make a naive implementation that calculates sample points one at a time, and then does the mlp shit with it
* make a better version that uses the tensor cores for mlp shit
* make an even better version that calculates sample points all at once via a big matrix (the way pytorch does it)
* figure out the neurex optimization shit and do that