### Setup

This project uses v3.0 of Vortex. The project structure should look like this:

```
parentfolder/
|--- vortex/
|--- nerf-renderer/
```

1. Create a parent folder.
2. Within that parent folder, git clone Vortex, git checkout 3.0, and then build it
3. Within that parent folder, git clone this repo

### Training NeRF

Within `nerf-renderer/` there is a folder called `ngp/`. This folder contains the code to train a NeRF on a set of images (in `ngp/novel_views`). Download the training and testing datasets from [here](https://drive.google.com/drive/folders/1eO7DXFhWWpauC-9LDhOimtIKxY3yRCIm) and place them under `nerf-renderer/ngp/`. (These training files are sourced from [here](https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/Instant_Neural_Graphics_Primitives_with_a_Multiresolution_Hash_Encoding) repo). 

Then:
* Create a python virtual environment (using `real_requirements.txt`)
* Make sure `SKIP_TRAIN` at the top of `ngp.py` is `False`
* run `ngp.py`. This will train a NeRF on the views in `novel_views/`. This takes some time (about 2 hours or so).

Once the NeRF is trained, it will produce a file called `ngp_weights.pt`. 

Next, you need to run `convert.py` to convert the testing dataset and the model weights into a C++-friendly file format. 

### vortex itself


To run the code:

1. cd into the `nerf-renderer` folder
2. run `make` 
3. run `make run-simx` to run the code