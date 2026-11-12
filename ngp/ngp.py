import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image

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
HN, HF = 2, 6
SKIP_TRAIN = False


@torch.no_grad()
def test(model, device, hn, hf, dataset, img_index, chunk_size=5, nb_bins=192, H=400, W=400):
    # dataset is a numpy memmap: only the slice we need is read into RAM
    sl = slice(img_index * H * W, (img_index + 1) * H * W)
    ray_origins = torch.from_numpy(np.array(dataset[sl, :3]))
    ray_directions = torch.from_numpy(np.array(dataset[sl, 3:6]))

    px_values = []  # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):  # iterate over chunks
        ro = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        rd = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        px_values.append(render_rays(model, ro, rd, hn=hn, hf=hf, nb_bins=nb_bins).cpu())
    img = torch.cat(px_values).numpy().reshape(H, W, 3)
    img = (img.clip(0, 1) * 255.).astype(np.uint8)
    Image.fromarray(img).save(f'novel_views/full_img_{img_index}.png')


class NGP(torch.nn.Module):

    def __init__(self, T, Nl, L, device, aabb_scale, F=2):
        super(NGP, self).__init__()
        self.T = T
        self.Nl = Nl
        self.F = F
        self.L = L  # For encoding directions
        self.aabb_scale = aabb_scale
        self.lookup_tables = torch.nn.ParameterDict(
            {str(i): torch.nn.Parameter((torch.rand(
                (T, 2), device=device) * 2 - 1) * 1e-4) for i in range(len(Nl))})
        self.pi1, self.pi2, self.pi3 = 1, 2_654_435_761, 805_459_861
        self.density_MLP = nn.Sequential(nn.Linear(self.F * len(Nl), 64),
                                         nn.ReLU(), nn.Linear(64, 16)).to(device)
        self.color_MLP = nn.Sequential(nn.Linear(27 + 16, 64), nn.ReLU(),
                                       nn.Linear(64, 64), nn.ReLU(),
                                       nn.Linear(64, 3), nn.Sigmoid()).to(device)

    def positional_encoding(self, x):
        out = [x]
        for j in range(self.L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        x = x / self.aabb_scale  # not in-place: don't mutate the caller's tensor
        # Indices of points inside the bounding box, computed once
        idx = (x.abs() < .5).all(dim=1).nonzero(as_tuple=True)[0]

        color = torch.zeros((x.shape[0], 3), device=x.device)
        if idx.numel() == 0:
            # nothing in the box: zero density everywhere (matches exp(-100000) == 0)
            return color, torch.zeros(x.shape[0], device=x.device)

        xm = x[idx] + 0.5   # masked points, in [0, 1]^3
        dm = d[idx]         # masked directions

        log_sigma = torch.full((x.shape[0],), -100000., device=x.device)
        features = torch.empty((xm.shape[0], self.F * len(self.Nl)), device=x.device)

        for i, N in enumerate(self.Nl):
            # Computing vertices, use nn.functional.grid_sample convention
            floor = torch.floor(xm * N)
            ceil = torch.ceil(xm * N)
            vertices = torch.zeros((xm.shape[0], 8, 3), dtype=torch.int64, device=x.device)
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat((ceil[:, 0, None], floor[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 2] = torch.cat((floor[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 4] = torch.cat((floor[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 6] = torch.cat((floor[:, 0, None], ceil[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 5] = torch.cat((ceil[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 3] = torch.cat((ceil[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 7] = ceil

            # hashing
            a = vertices[:, :, 0] * self.pi1
            b = vertices[:, :, 1] * self.pi2
            c = vertices[:, :, 2] * self.pi3
            h_x = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a, b), c), self.T)

            # Lookup
            looked_up = self.lookup_tables[str(i)][h_x].transpose(-1, -2)
            volume = looked_up.reshape((looked_up.shape[0], 2, 2, 2, 2))
            features[:, i * 2:(i + 1) * 2] = torch.nn.functional.grid_sample(
                volume,
                ((xm * N - floor) - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1),
                align_corners=False,  # same as the old default, just silences the warning
            ).squeeze(-1).squeeze(-1).squeeze(-1)

        xi = self.positional_encoding(dm)
        h = self.density_MLP(features)
        log_sigma[idx] = h[:, 0]
        color[idx] = self.color_MLP(torch.cat((h, xi), dim=1))
        return color, torch.exp(log_sigma)


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor(
        [1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    alpha = 1 - torch.exp(-sigma.reshape(x.shape[:-1]) * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors.reshape(x.shape)).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, dataset, device='cpu', hn=0, hf=1, nb_epochs=10,
          nb_bins=192, batch_size=2 ** 14, max_steps=None):
    # No DataLoader: its shuffling sampler builds a Python list of every ray index
    # (~64M ints), which blew up host RAM. A randperm tensor is far cheaper.
    n = dataset.shape[0]
    step = 0
    for _ in range(nb_epochs):
        perm = torch.randperm(n)
        total = (n + batch_size - 1) // batch_size
        if max_steps is not None:
            total = min(total, max_steps - step)
        bar = tqdm(range(total))
        for k in bar:
            s = k * batch_size
            batch = dataset[perm[s:s + batch_size]]
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            gt_px_values = batch[:, 6:].to(device)
            pred_px_values = render_rays(nerf_model, ray_origins, ray_directions,
                                         hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((gt_px_values - pred_px_values) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if k % 50 == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")
        if max_steps is not None and step >= max_steps:
            break


if __name__ == "__main__":
    # import torch
    # sd = torch.load('ngp_weights.pt', map_location='cpu')
    # for k, v in sd.items():
    #     print(k, tuple(v.shape))

    # exit(0)

    device = 'cuda'
    print("start")

    # Only load the 2 GB training set if we're actually training
    training_dataset = None
    if not SKIP_TRAIN:
        training_dataset = torch.from_numpy(np.load('training_data_800x800.pkl', allow_pickle=True))
        print("training loaded", tuple(training_dataset.shape))
    # Test set stays on disk as a memmap
    testing_dataset = np.load('testing_data_800x800.pkl', allow_pickle=True, mmap_mode='r')
    print("testing mapped", testing_dataset.shape)

    L = 16
    F = 2
    T = 2 ** 19
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b ** l)) for l in range(L)]
    model = NGP(T, Nl, 4, device, 3)
    model_optimizer = torch.optim.Adam(
        [{"params": model.lookup_tables.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 0.},
         {"params": model.density_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10 ** -6},
         {"params": model.color_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10 ** -6}])

    if SKIP_TRAIN:
        print("loading ngp_weights.pt")
        model.load_state_dict(torch.load('ngp_weights.pt', map_location=device))
    else:
        print("starting training")
        train(model, model_optimizer, training_dataset, nb_epochs=1, device=device,
              hn=HN, hf=HF, nb_bins=NB_BINS, batch_size=BATCH_SIZE, max_steps=MAX_STEPS)
        print("ending training")
        torch.save(model.state_dict(), 'ngp_weights.pt')

    os.makedirs('novel_views', exist_ok=True)
    for img_index in tqdm(range(TEST_IMAGES)):
        test(model, device, HN, HF, testing_dataset, img_index,
             chunk_size=TEST_CHUNK_ROWS, nb_bins=NB_BINS, H=H, W=W)
        torch.cuda.empty_cache()   # release cached VRAM between images
    print("done: see novel_views/")