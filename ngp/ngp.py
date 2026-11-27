import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from torch import Tensor

# adapted from https://papers-100-lines.medium.com/instant-3d-reconstruction-with-python-and-instant-ngp-68973cd6f706
# the original didnt work and was stuck in dependency hell so an llm rewrite it to make it work and not shit

# settings sized for my lapto p with 4gb vram. initial paremters were set low and then
# i increased them until it looked good and also didnt crash
BATCH_SIZE = 2 ** 11      # rays per training step
NB_BINS = 192             # samples per ray (original was 192)
MAX_STEPS = None          # stop training after this many steps (None = full epoch, 2 minutes takes like 5 minutes or something)
TEST_IMAGES = 2           # number of novel views to render (original: 200)
TEST_CHUNK_ROWS = 1       # image rows rendered at once (5 * 800 = 4000 rays)
H, W = 800, 800
HN, HF = 2, 6             # distance to the near plane and far plane
SKIP_TRAIN = True

class NGP(torch.nn.Module):
    def __init__(self, numRows: int, gridResolutions: list[int], frequencies: int, device: str, aabbScale: int, featuresPerEntry: int = 2):
        super(NGP, self).__init__()

        # the number of rows in the hash table (2^19 apparently, in the instant ngp paper)
        self.hashTableRows: int = numRows

        # list with the grid resolution of each level (16 to 2048). len(gridResolutions) is the number of levels (16)
        # so this stores [16, 22, 30, 42, 58, ..., 1072, 1481, 2048]
        self.gridResolutions: list[int] = gridResolutions
        
        # number of features per hash table entry (2). so the hash table is hashTableRows rows by featuresPerEntry columns (2^19  by 2)
        self.featuresPerEntry: int = featuresPerEntry
        
        # number of frequencies used to encode viewing direction (see positional_encoding)
        self.numFrequencies: int = frequencies 
        
        # aabbScale = 3 means the scene fits in a 3x3x3 cube. in forward(), it uses this to convert to a unit cube ((x / aabb scale) + 0.5)
        self.aabbScale: int = aabbScale

        # create lookup tables with wierd pytorch syntax. 
        # use torch.rand values in [0, 1], then bring them into the range [-1, 1], then scale by 1e-4. apparently instant-ngp does this.
        # this wacky syntax lets it do it all in one go i guess
        self.lookupTables: torch.nn.ParameterDict = torch.nn.ParameterDict(
            {str(i): torch.nn.Parameter((torch.rand((numRows, 2), device=device) * 2 - 1) * 1e-4) for i in range(len(gridResolutions))}
        )

        # the three random primes for hashing
        self.prime1: int = 1
        self.prime2: int = 2_654_435_761
        self.prime3: int = 805_459_861

        # the two neural networks for density and color
        self.density_MLP: nn.Sequential = nn.Sequential(nn.Linear(self.featuresPerEntry * len(gridResolutions), 64),
                                          nn.ReLU(), 
                                          nn.Linear(64, 16)).to(device)
        self.color_MLP: nn.Sequential = nn.Sequential(nn.Linear(27 + 16, 64), 
                                        nn.ReLU(),
                                        nn.Linear(64, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 3), 
                                        nn.Sigmoid()).to(device)

    # poor man's spherical harmonics
    # the color of a point depends on the direction you view it from. in gaussian splats/instant ngp, 
    # you use spherical harmonics. here, we use a simpler version. (numFrequencies = 4)
    # the positional encoding becomes [x, sin(x), cos(x), sin(2x), cos(2x), sin(4x), cos(4x), sin(8x), cos(8x)]
    def positional_encoding(self, x: Tensor) -> Tensor:
        out: list[Tensor] = [x]
        for i in range(self.numFrequencies):
            out.append(torch.sin(2**i * x))
            out.append(torch.cos(2**i * x))

        return torch.cat(out, dim=1)

    # x is a tensor of size (BxN, 3) where B is the number of rays, and N is samples per ray (3 for xyz)
    # in render_rays, it builds a tensor of (B rays) by (N samples per ray) by 3. this gets flattened into (BxN, 3), ie a giant list of sample points.
    # -----
    # d is just the ray direction for each point. for example:
    #
    # x (6 rows)                   d (6 rows)
    # ray 0, sample 0: (x,y,z)     ray 0's direction
    # ray 0, sample 1: (x,y,z)     ray 0's direction
    # ray 0, sample 2: (x,y,z)     ray 0's direction
    # ray 1, sample 0: (x,y,z)     ray 1's direction
    # ray 1, sample 1: (x,y,z)     ray 1's direction
    # ray 1, sample 2: (x,y,z)     ray 1's direction
    def forward(self, x: Tensor, d: Tensor):
        # scale x by aabbScale, to bring the whole scene into the range [-0.5, 0.5]^3 (not in-place to avoid mutating the caller's tensor)
        x = x / self.aabbScale

        # weird pytorch syntax to get a list of the indices of all points inside [-0.5, 0.5]^3
        idx: Tensor = (x.abs() < .5).all(dim=1).nonzero(as_tuple=True)[0]

        # if all points are outside the box, return 0 for color and density
        color: Tensor = torch.zeros((x.shape[0], 3), device=x.device)
        if idx.numel() == 0:
            return color, torch.zeros(x.shape[0], device=x.device)

        xm: Tensor = x[idx] + 0.5   # all points inside the box, shifted to be in [0, 1]^3
        dm: Tensor = d[idx]         # viewing directions for all points in the box

        # 1D tensor of length (num rays x num samples per ray), with all values set to -100,000 (because e^-100000 is 0 in floating piont. the last line of this 
        # function does torch.exp(log_sigma)). so untouched values = values with 0 density = stay 0 
        log_sigma: Tensor = torch.full((x.shape[0],), -100000., device=x.device)

        # 2D tensor of (K, 32) where K is (num rays x samples per ray) and 32 is because (2 features per entry * 16 grid resolutions)
        # so the tensor is (sample point of a ray, every single grid resolution entyr for that point)
        features: Tensor = torch.empty((xm.shape[0], self.featuresPerEntry * len(self.gridResolutions)), device=x.device)

        # loop over each resolution, and figure out all sample points in a given resolution at once
        for i, resolution in enumerate(self.gridResolutions):
            # figure out the corners for each point's box
            floor: Tensor = torch.floor(xm * resolution)
            ceil: Tensor = torch.ceil(xm * resolution)

            # get the integer coordinates for all 8 corners
            vertices: Tensor = torch.zeros((xm.shape[0], 8, 3), dtype=torch.int64, device=x.device)
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat(( ceil[:, 0, None], floor[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 2] = torch.cat((floor[:, 0, None], ceil [:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 3] = torch.cat(( ceil[:, 0, None], ceil [:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 4] = torch.cat((floor[:, 0, None], floor[:, 1, None], ceil [:, 2, None]), dim=1)
            vertices[:, 5] = torch.cat(( ceil[:, 0, None], floor[:, 1, None], ceil [:, 2, None]), dim=1)
            vertices[:, 6] = torch.cat((floor[:, 0, None], ceil [:, 1, None], ceil [:, 2, None]), dim=1)
            vertices[:, 7] = ceil

            # hash each corner and get its index into the lookup table
            a: Tensor = vertices[:, :, 0] * self.prime1
            b: Tensor = vertices[:, :, 1] * self.prime2
            c: Tensor = vertices[:, :, 2] * self.prime3
            h_x: Tensor = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a, b), c), self.hashTableRows)

            # Lookup
            # (num points, 8 corner point indices, 2 features per corner), which is then reshaped into (num points, 2 features, 8 corner indices)
            looked_up: Tensor = self.lookupTables[str(i)][h_x].transpose(-1, -2)

            # reshape into (num points, 2 features, 2 z indices, 2 y indices, 2 x indices)
            volume: Tensor = looked_up.reshape((looked_up.shape[0], 2, 2, 2, 2))

            # the final feature. abuses pytorch's grid_sample function to do trilinear interpolation for each point's grid corners
            features[:, i * 2:(i + 1) * 2] = torch.nn.functional.grid_sample(
                volume,
                ((xm * resolution - floor) - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1),
                align_corners=False,  # same as the old default, just silences the warning
            ).squeeze(-1).squeeze(-1).squeeze(-1)

        # get the frequency encoding for each position (so it can learn view dependent color i guess)
        xi: Tensor = self.positional_encoding(dm)

        # put the sample points and 2 features per grid resolution into the density mlp, which returns a (num points, 16) tensor
        # where col 0 is the log of the desnity, and col 1-15 is some bullshit numbers representing the local geometry/density/whatever
        h: Tensor = self.density_MLP(features)

        # store the log densities in log_sigma
        log_sigma[idx] = h[:, 0]

        # put the density tensor and the frequency encoding through the color mlp, which returns a predicted rgb color in the range [0, 1]
        # store these in color 
        color[idx] = self.color_MLP(torch.cat((h, xi), dim=1))

        # return the predicted colors and predicted log density for this pass
        return color, torch.exp(log_sigma)


@torch.no_grad()
def test(model: NGP, device: str, hn, hf, dataset, img_index, chunk_size=5, nb_bins=192, H=400, W=400):
    # dataset is a numpy memmap: only the slice we need is read into RAM
    sl = slice(img_index * H * W, (img_index + 1) * H * W)
    ray_origins = torch.from_numpy(np.array(dataset[sl, :3]))
    ray_directions = torch.from_numpy(np.array(dataset[sl, 3:6]))

    px_values = []  # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):  # iterate over chunks
        ro = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        rd = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        px_values.append(render_rays(model, ro, rd, near=hn, far=hf, bins=nb_bins).cpu())
    img = torch.cat(px_values).numpy().reshape(H, W, 3)
    img = (img.clip(0, 1) * 255.).astype(np.uint8)
    Image.fromarray(img).save(f'novel_views/full_img_{img_index}.png')



def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerfModel: NGP, rayOrigins: Tensor, rayDirections: Tensor, near, far, bins: int = 192):
    # generate 192 sample points between the near and far planes
    device = rayOrigins.device
    t: Tensor = torch.linspace(near, far, bins, device=device).expand(rayOrigins.shape[0], bins)
    
    # some pytorch bullshit to get a list of intervals
    mid: Tensor = (t[:, :-1] + t[:, 1:]) / 2.
    lower: Tensor = torch.cat((t[:, :1], mid), -1)
    upper: Tensor = torch.cat((mid, t[:, -1:]), -1)

    # u is some random jitter within that interval
    u: Tensor = torch.rand(t.shape, device=device)
    
    # pick a random sample within each interval
    t = lower + (upper - lower) * u  

    # subtracts each sample with the next sample, to get a tensor that represents distance from one sample to the next
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(rayOrigins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x: Tensor = rayOrigins.unsqueeze(1) + t.unsqueeze(2) * rayDirections.unsqueeze(1)

    # some tensor reshaping bullshit idk
    rayDirections = rayDirections.expand(bins, rayDirections.shape[0], 3).transpose(0, 1)

    # builds a tensor of (B rays) by (N samples per ray) by 3. this gets flattened into (BxN, 3), ie a giant list of sample points
    # then passes that into the NGP model to render it out and get the colors and density at each sample point
    # pytorch bullshit: nerfModel is an NGP object, but if you call it like a function, it treats like a call to .forward()
    colors, sigma = nerfModel(x.reshape(-1, 3), rayDirections.reshape(-1, 3))

    # calculate the opacity at each sample point
    alpha: Tensor = 1 - torch.exp(-sigma.reshape(x.shape[:-1]) * delta)

    # calculate the transmittence per sample point
    weights: Tensor = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # converts it back from (BxN, 3) into (B rays, N samples per ray, 3)
    # multiplies predicted color at each point by the transmittence at each point, 
    # then adds it up to get the final color per ray, so the final shape is (B, 3)
    c: Tensor = (weights * colors.reshape(x.shape)).sum(dim=1)

    # Add up the transmittences per point along the ray into one final transmittence per ray (shape is a 1d array of length num rays)
    weight_sum = weights.sum(-1).sum(-1)

    # the image background was white, so rays that miss or are transparent need to be mixed with white. 
    return c + 1 - weight_sum.unsqueeze(-1)


# dataset here is a tensor where each row is [origin xyz, direction xyz, true color xyz]
def train(nerfModel: NGP, optimizer: torch.optim.Adam, dataset: Tensor, device='cpu', hn=0, hf=1, epochs=10, bins=192, batchSize=2**14, maxSteps=None):
    # No DataLoader: its shuffling sampler builds a Python list of every ray index (~64M ints), which blows up RAM. A randperm tensor is far cheaper.
    n: int = dataset.shape[0] # number of training rays
    step: int = 0
    for _ in range(epochs):
        randomRays: Tensor = torch.randperm(n) # get a random combination of ray indices

        # calculate the batch size
        total: int = (n + batchSize - 1) // batchSize
        if maxSteps is not None:
            total = min(total, maxSteps - step)
            
        bar: tqdm = tqdm(range(total))
        for k in bar:
            s: int = k * batchSize # offset into randomRays
            batch: Tensor = dataset[randomRays[s:s + batchSize]] # returns just those rays out of the dataset

            # extract the ray origin, direction, and color from that
            rayOrigins: Tensor = batch[:, :3].to(device)
            rayDirections: Tensor = batch[:, 3:6].to(device)
            groundTruthColor: Tensor = batch[:, 6:].to(device)

            # render the ray and get the predicted values
            predictedValues: Tensor = render_rays(nerfModel, rayOrigins, rayDirections, near=hn, far=hf, bins=bins)

            # measure loss, do gradient descent stuff
            loss: Tensor = ((groundTruthColor - predictedValues) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            if k % 50 == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")
                
        if maxSteps is not None and step >= maxSteps: break


if __name__ == "__main__":
    # import torch
    # sd = torch.load('ngp_weights.pt', map_location='cpu')
    # for k, v in sd.items():
    #     print(k, tuple(v.shape))

    # exit(0)

    device = 'cuda'
    print("start")

    # Only load the 2 GB training set if we're actually training
    training_dataset: Tensor = Tensor()
    if not SKIP_TRAIN:
        training_dataset = torch.from_numpy(np.load('training_data_800x800.pkl', allow_pickle=True))
        print("training loaded", tuple(training_dataset.shape))
    # Test set stays on disk as a memmap
    testing_dataset: Tensor = np.load('testing_data_800x800.pkl', allow_pickle=True, mmap_mode='r')
    print("testing mapped", testing_dataset.shape)

    L = 16
    F = 2
    numRows: int = 2 ** 19
    N_min: int = 16
    N_max: int = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    gridResolution = [int(np.floor(N_min * b ** l)) for l in range(L)]
    model: NGP = NGP(numRows, gridResolution, 4, device, 3)
    modelOptimizer: torch.optim.Adam = torch.optim.Adam(
        [{"params": model.lookupTables.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 0.},
         {"params": model.density_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10 ** -6},
         {"params": model.color_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10 ** -6}])

    if SKIP_TRAIN:
        print("loading ngp_weights.pt")
        model.load_state_dict(torch.load('ngp_weights.pt', map_location=device))
    else:
        print("starting training")
        train(model, modelOptimizer, training_dataset, epochs=1, device=device, hn=HN, hf=HF, bins=NB_BINS, batchSize=BATCH_SIZE, maxSteps=MAX_STEPS)
        print("ending training")
        torch.save(model.state_dict(), 'ngp_weights.pt')

    # os.makedirs('novel_views', exist_ok=True)
    # for img_index in tqdm(range(TEST_IMAGES)):
    #     test(model, device, HN, HF, testing_dataset, img_index, chunk_size=TEST_CHUNK_ROWS, nb_bins=NB_BINS, H=H, W=W)
    #     torch.cuda.empty_cache()   # release cached VRAM between images
    test(model, device, HN, HF, testing_dataset, 30, chunk_size=TEST_CHUNK_ROWS, nb_bins=NB_BINS, H=H, W=W)
    print("done: see novel_views/")