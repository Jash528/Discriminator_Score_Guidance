# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
import subprocess, sys, os

os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':' + os.environ.get('PATH','')

pkgs = [
    'numpy<2',                   
    'diffusers==0.30.3',         
    'accelerate==0.30.1',        
    'transformers==4.40.0',      
    'datasets==2.19.0',          
    'huggingface_hub==0.23.4',   
    'clean-fid',
    'torchmetrics',
    'matplotlib',
    'scipy',
    'Pillow',
]
print("Installing packages (this takes ~2 min)...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       '--user', '--quiet', '--force-reinstall'] + pkgs)
print("Done! Now click: Kernel → Restart Kernel, then run Cell 2 onwards.")

# %% [code] Cell 2
import os, sys, json, numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

print(f"torch      : {torch.__version__}")
print(f"CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

OUT = os.path.expanduser("~/gan_outputs")
for d in [OUT,
          f"{OUT}/samples/unguided", f"{OUT}/samples/dsg",
          f"{OUT}/data/cifar10_real"]:
    os.makedirs(d, exist_ok=True)
print(f"Output dir : {OUT}")

# %% [code] Cell 3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainset = torchvision.datasets.CIFAR10(
    root=f'{OUT}/data', train=True, download=True, transform=transform
)
loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=(device=="cuda")
)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz,    ngf*4, 4,1,0, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4,2,1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4,2,1, bias=False), nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   3,     4,2,1, bias=False), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,      ndf,   4,2,1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,    ndf*2, 4,2,1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2,  ndf*4, 4,2,1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4,  ndf*8, 4,2,1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8,  1,     2,1,0, bias=False), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).view(-1,1)

NZ, N_EPOCHS = 100, 100
G = Generator(NZ).to(device)
D = Discriminator().to(device)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
G.apply(weights_init); D.apply(weights_init)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
bce   = nn.BCELoss()

print(f"Training DCGAN for {N_EPOCHS} epochs on {device}...")
for epoch in range(N_EPOCHS):
    d_losses, g_losses = [], []
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device); b = real_imgs.size(0)
        D.zero_grad()
        label = torch.full((b,1), 1.0, device=device)
        loss_real = bce(D(real_imgs), label); loss_real.backward()
        noise = torch.randn(b, NZ, 1, 1, device=device); fake = G(noise)
        label.fill_(0.0); loss_fake = bce(D(fake.detach()), label); loss_fake.backward()
        opt_D.step()
        G.zero_grad(); label.fill_(1.0)
        loss_G = bce(D(fake), label); loss_G.backward(); opt_G.step()
        d_losses.append((loss_real+loss_fake).item()); g_losses.append(loss_G.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{N_EPOCHS} | D: {sum(d_losses)/len(d_losses):.4f} | G: {sum(g_losses)/len(g_losses):.4f}")

torch.save(D.state_dict(), f'{OUT}/dcgan_discriminator_cifar10.pt')
torch.save(G.state_dict(), f'{OUT}/dcgan_generator_cifar10.pt')
print("Checkpoints saved.")

# %% [code] Cell 4
from diffusers import DDPMPipeline
from cleanfid import fid as clean_fid

print("Loading DDPM pipeline...")
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
D_fid = Discriminator().to(device)
D_fid.load_state_dict(torch.load(f'{OUT}/dcgan_discriminator_cifar10.pt', map_location=device))
D_fid.eval()

DELTA     = 0.1
N_SAMPLES = 50000
BATCH     = 32 

print("Saving real CIFAR-10 images...")
real_set = torchvision.datasets.CIFAR10(
    root=f'{OUT}/data', train=True, download=False,
    transform=torchvision.transforms.ToTensor()
)
for i, (img, _) in enumerate(real_set):
    if i >= N_SAMPLES: break
    torchvision.utils.save_image(img, f'{OUT}/data/cifar10_real/{i:05d}.png')
print(f"Saved {N_SAMPLES} real images.")

print(f"Generating {N_SAMPLES} unguided samples...")
n_done = 0
while n_done < N_SAMPLES:
    b = min(BATCH, N_SAMPLES - n_done)
    with torch.no_grad():
        out = pipe(batch_size=b, num_inference_steps=1000)
    for j, img in enumerate(out.images):
        img.save(f'{OUT}/samples/unguided/{n_done+j:05d}.png')
    n_done += b
    if n_done % 5000 == 0: print(f"  {n_done}/{N_SAMPLES}")
print("Unguided sampling done.")

def dsg_grad(D, x_t, scheduler, noise_pred, t):
    alpha_bar = scheduler.alphas_cumprod[t]
    x0_hat = (x_t - (1-alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
    x0_hat = x0_hat.clamp(-1,1).detach().requires_grad_(True)
    d_out    = D(x0_hat).clamp(DELTA, 1-DELTA)
    log_odds = (torch.log(d_out) - torch.log(1-d_out)).sum()
    log_odds.backward()
    return x0_hat.grad.detach()

def dsg_sample_batch(pipe, D, batch_size, gamma=0.5, timestep_cutoff=(0.2,0.8)):
    unet = pipe.unet; scheduler = pipe.scheduler
    scheduler.set_timesteps(1000); T = len(scheduler.timesteps)
    x_t = torch.randn(batch_size, 3, 32, 32).to(device)
    for i, t in enumerate(scheduler.timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = unet(x_t, t_batch).sample
        x_prev = scheduler.step(noise_pred, t, x_t).prev_sample
        progress = i / T
        if timestep_cutoff[0] < progress < timestep_cutoff[1]:
            grad      = dsg_grad(D, x_t, scheduler, noise_pred, t)
            grad_flat = grad.view(batch_size,-1)
            grad_norm = grad_flat.norm(dim=1,keepdim=True).clamp(min=1e-8)
            grad_n    = (grad_flat/grad_norm).view_as(grad)
            x_norm    = x_t.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
            x_prev    = x_prev + gamma * x_norm * grad_n
        x_t = x_prev.clamp(-1,1)
    imgs = ((x_t.cpu()+1)/2).clamp(0,1)
    return [torchvision.transforms.ToPILImage()(imgs[j]) for j in range(batch_size)]

print(f"Generating {N_SAMPLES} DSG-guided samples...")
n_done = 0
while n_done < N_SAMPLES:
    b = min(BATCH, N_SAMPLES - n_done)
    imgs = dsg_sample_batch(pipe, D_fid, b, gamma=0.5)
    for j, img in enumerate(imgs): img.save(f'{OUT}/samples/dsg/{n_done+j:05d}.png')
    n_done += b
    if n_done % 5000 == 0: print(f"  {n_done}/{N_SAMPLES}")
print("DSG sampling done.")

print("Computing FID (this takes a few minutes)...")
fid_unguided = clean_fid.compute_fid(f'{OUT}/samples/unguided/', f'{OUT}/data/cifar10_real/')
fid_dsg      = clean_fid.compute_fid(f'{OUT}/samples/dsg/',      f'{OUT}/data/cifar10_real/')
print(f"Unguided FID : {fid_unguided:.2f}")
print(f"DSG FID      : {fid_dsg:.2f}")

# %% [code] Cell 5
GAMMAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
N_ABL  = 5000
gamma_results = {}

print(f"Gamma ablation ({N_ABL} samples each)...")
for gamma in GAMMAS:
    out_dir = f'{OUT}/samples/ablation_gamma_{gamma}'
    os.makedirs(out_dir, exist_ok=True)
    n_done = 0
    while n_done < N_ABL:
        b = min(BATCH, N_ABL - n_done)
        if gamma == 0.0:
            with torch.no_grad(): imgs = pipe(batch_size=b, num_inference_steps=1000).images
        else:
            imgs = dsg_sample_batch(pipe, D_fid, b, gamma=gamma)
        for j, img in enumerate(imgs): img.save(f'{out_dir}/{n_done+j:05d}.png')
        n_done += b
    score = clean_fid.compute_fid(out_dir, f'{OUT}/data/cifar10_real/')
    gamma_results[gamma] = score
    print(f"  gamma={gamma:.1f} | FID={score:.2f}")

json.dump({str(k):v for k,v in gamma_results.items()}, open(f'{OUT}/results_ablation_gamma.json','w'))
best_gamma = min(gamma_results, key=gamma_results.get)

plt.figure(figsize=(7,5))
plt.plot(list(gamma_results.keys()), list(gamma_results.values()), 'bo-', lw=2, ms=8)
plt.axhline(gamma_results[0.0], color='red', ls='--', lw=1.5, label=f'Unguided (FID={gamma_results[0.0]:.1f})')
plt.xlabel('γ', fontsize=13); plt.ylabel('FID ↓', fontsize=13)
plt.title('DSG Ablation — FID vs γ', fontsize=14, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{OUT}/figure3_gamma_ablation.pdf', dpi=300, bbox_inches='tight'); plt.close()
print(f"Best gamma: {best_gamma}  →  figure3 saved")

# %% [code] Cell 6
CUTOFFS = [(0.1,0.4),(0.2,0.6),(0.3,0.7),(0.2,0.8),(0.1,0.9)]
cutoff_results = {}

print("Cutoff ablation...")
for low, high in CUTOFFS:
    label   = f"{low}-{high}"
    out_dir = f'{OUT}/samples/ablation_cutoff_{label}'
    os.makedirs(out_dir, exist_ok=True)
    n_done = 0
    while n_done < N_ABL:
        b = min(BATCH, N_ABL - n_done)
        imgs = dsg_sample_batch(pipe, D_fid, b, gamma=best_gamma, timestep_cutoff=(low,high))
        for j, img in enumerate(imgs): img.save(f'{out_dir}/{n_done+j:05d}.png')
        n_done += b
    score = clean_fid.compute_fid(out_dir, f'{OUT}/data/cifar10_real/')
    cutoff_results[label] = score
    print(f"  cutoff={label} | FID={score:.2f}")

json.dump(cutoff_results, open(f'{OUT}/results_ablation_cutoff.json','w'))
best_cutoff = min(cutoff_results, key=cutoff_results.get)

plt.figure(figsize=(7,5))
plt.bar(list(cutoff_results.keys()), list(cutoff_results.values()), color='steelblue', edgecolor='white', width=0.5)
plt.axhline(gamma_results[0.0], color='red', ls='--', lw=1.5, label=f'Unguided (FID={gamma_results[0.0]:.1f})')
plt.xlabel('Timestep window', fontsize=13); plt.ylabel('FID ↓', fontsize=13)
plt.title('DSG Ablation — FID vs Timestep Cutoff', fontsize=14, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig(f'{OUT}/figure4_cutoff_ablation.pdf', dpi=300, bbox_inches='tight'); plt.close()
print("figure4 saved")

# %% [code] Cell 7
from datasets import load_dataset
from torchvision import transforms as T

print("Loading CelebA from HuggingFace (~1.4 GB first run)...")
celeba = load_dataset(
    "flwrlabs/celeba", split="train[:20000]",
    cache_dir=os.path.expanduser("~/.cache/huggingface")
)
print(f"Loaded {len(celeba)} images")

os.makedirs(f"{OUT}/samples/celeba_unguided", exist_ok=True)
os.makedirs(f"{OUT}/samples/celeba_dsg",      exist_ok=True)

class AttrDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   64,  4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4,2,1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(512*4*4, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

attr_tf = T.Compose([T.Resize((64,64)), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
ATTR     = "Smiling"
ATTR_IDX = celeba.features['attributes'].feature.names.index(ATTR)

print(f"Building {ATTR} dataset...")
pos_imgs, neg_imgs = [], []
for item in celeba:
    img = attr_tf(item['image'].convert('RGB'))
    (pos_imgs if item['attributes'][ATTR_IDX]==1 else neg_imgs).append(img)
    if len(pos_imgs)>=3000 and len(neg_imgs)>=3000: break
pos_imgs = torch.stack(pos_imgs[:3000]); neg_imgs = torch.stack(neg_imgs[:3000])
print(f"Pos: {len(pos_imgs)}  Neg: {len(neg_imgs)}")

D_attr = AttrDiscriminator().to(device)
opt_a  = torch.optim.Adam(D_attr.parameters(), lr=1e-4, betas=(0.5,0.999))
bce_a  = nn.BCELoss()

print("Training attribute discriminator (50 epochs)...")
for epoch in range(50):
    idx = torch.randperm(3000)
    for start in range(0, 3000, 64):
        bi  = idx[start:start+64]
        pos = pos_imgs[bi].to(device); neg = neg_imgs[bi].to(device)
        n   = min(len(pos),len(neg));  pos,neg = pos[:n],neg[:n]
        loss = bce_a(D_attr(pos), torch.ones(n,1,device=device)) +                bce_a(D_attr(neg), torch.zeros(n,1,device=device))
        opt_a.zero_grad(); loss.backward(); opt_a.step()
    if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}/50 | loss={loss.item():.4f}")

D_attr.eval()
torch.save(D_attr.state_dict(), f'{OUT}/celeba_attr_discriminator.pt')
print("Attribute discriminator saved.")

# %% [code] Cell 8
from diffusers import DDPMPipeline

print("Loading CelebA DDPM (google/ddpm-celebahq-256)...")
celeba_pipe = DDPMPipeline.from_pretrained(
    "google/ddpm-celebahq-256",
    cache_dir=os.path.expanduser("~/.cache/huggingface")
).to(device)
print("Loaded.")

def dsg_grad_celeba(D, x_t, scheduler, noise_pred, t):
    alpha_bar = scheduler.alphas_cumprod[t]
    x0_hat    = (x_t - (1-alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
    x0_hat    = x0_hat.clamp(-1,1).detach().requires_grad_(True)
    x0_64     = torch.nn.functional.interpolate(x0_hat, size=(64,64), mode='bilinear', align_corners=False)
    d_out     = D(x0_64).clamp(DELTA, 1-DELTA)
    (torch.log(d_out) - torch.log(1-d_out)).sum().backward()
    return x0_hat.grad.detach()

def dsg_sample_celeba(pipe, D, n_samples=4, gamma=0.5, timestep_cutoff=(0.2,0.8), img_size=256):
    unet = pipe.unet; scheduler = pipe.scheduler
    scheduler.set_timesteps(1000); T = len(scheduler.timesteps)
    x_t = torch.randn(n_samples, 3, img_size, img_size, device=device)
    for i, t in enumerate(scheduler.timesteps):
        t_b = torch.full((n_samples,), t, device=device, dtype=torch.long)
        with torch.no_grad(): noise_pred = unet(x_t, t_b).sample
        x_prev = scheduler.step(noise_pred, t, x_t).prev_sample
        if timestep_cutoff[0] < i/T < timestep_cutoff[1]:
            grad      = dsg_grad_celeba(D, x_t, scheduler, noise_pred, t)
            grad_flat = grad.view(n_samples,-1)
            grad_n    = (grad_flat/grad_flat.norm(dim=1,keepdim=True).clamp(1e-8)).view_as(grad)
            x_norm    = x_t.view(n_samples,-1).norm(dim=1).view(n_samples,1,1,1)
            x_prev    = x_prev + gamma * x_norm * grad_n
        x_t = x_prev.clamp(-1,1)
    imgs = ((x_t.cpu()+1)/2).clamp(0,1)
    return [torchvision.transforms.ToPILImage()(imgs[j]) for j in range(n_samples)]

print("Generating unguided CelebA faces...")
with torch.no_grad():
    celeba_unguided = celeba_pipe(batch_size=4, num_inference_steps=1000)

print("Generating DSG smile-guided faces...")
celeba_guided = dsg_sample_celeba(celeba_pipe, D_attr, n_samples=4)

fig, axes = plt.subplots(2, 4, figsize=(14,8))
for col in range(4):
    axes[0,col].imshow(celeba_unguided.images[col]); axes[0,col].axis('off'); axes[0,col].set_title(f'Sample {col+1}',fontsize=10)
    axes[1,col].imshow(celeba_guided[col]);           axes[1,col].axis('off')
axes[0,0].set_ylabel("Unguided",fontsize=11,fontweight='bold',rotation=0,ha='right',va='center',labelpad=60)
axes[1,0].set_ylabel(f"DSG:{ATTR}",fontsize=11,fontweight='bold',rotation=0,ha='right',va='center',labelpad=60)
plt.suptitle(f'DSG Attribute Guidance — CelebA\nTarget: {ATTR}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/figure5_celeba_guidance.pdf', dpi=200, bbox_inches='tight'); plt.close()
print("figure5 saved.")

from transformers import CLIPProcessor, CLIPModel
print("Loading CLIP...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_score(images, prompt):
    inputs = clip_processor(text=[prompt], images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
        ie  = out.image_embeds / out.image_embeds.norm(dim=-1,keepdim=True)
        te  = out.text_embeds  / out.text_embeds.norm(dim=-1,keepdim=True)
    return (ie @ te.T).squeeze().mean().item()

prompt        = "a person with a big smile"
clip_unguided = clip_score(celeba_unguided.images, prompt)
clip_guided   = clip_score(celeba_guided,           prompt)
print(f"\nCLIP — prompt: '{prompt}'")
print(f"  Unguided : {clip_unguided:.4f}")
print(f"  DSG      : {clip_guided:.4f}")
print(f"  Δ        : +{clip_guided-clip_unguided:.4f}")

# %% [code] Cell 9
g_res = json.load(open(f'{OUT}/results_ablation_gamma.json'))
c_res = json.load(open(f'{OUT}/results_ablation_cutoff.json'))
best_g = min(g_res, key=g_res.get)
best_c = min(c_res, key=c_res.get)

print("="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\nTable 1 — CIFAR-10 FID")
print(f"  Unguided : {fid_unguided:.2f}")
print(f"  DSG      : {fid_dsg:.2f}  (Δ = {fid_unguided-fid_dsg:.2f})")
print(f"\nTable 2 — Gamma Ablation")
for g,f in sorted(g_res.items(), key=lambda x: float(x[0])):
    print(f"  gamma={float(g):.1f} : FID={f:.2f}{'  ← best' if g==best_g else ''}")
print(f"\nTable 3 — Cutoff Ablation")
for c,f in c_res.items():
    print(f"  {c} : FID={f:.2f}{'  ← best' if c==best_c else ''}")
print(f"\nTable 4 — CelebA CLIP ({ATTR})")
print(f"  Unguided : {clip_unguided:.4f}")
print(f"  DSG      : {clip_guided:.4f}")
print("="*60)
print(f"\nAll files in: {OUT}/")
for f in ['figure3_gamma_ablation.pdf','figure4_cutoff_ablation.pdf',
          'figure5_celeba_guidance.pdf','results_ablation_gamma.json',
          'results_ablation_cutoff.json','dcgan_discriminator_cifar10.pt',
          'dcgan_generator_cifar10.pt','celeba_attr_discriminator.pt']:
    exists = "✓" if os.path.exists(f'{OUT}/{f}') else "✗ MISSING"
    print(f"  {exists}  {f}")
