# Discriminator Score Guidance (DSG)

**Bridging GANs and Diffusion Models via the Score Function**

A research project exploring whether a pre-trained GAN discriminator 
can serve as a zero-shot guidance signal for DDPM sampling.

## Milestone Demo Results

- Theorem validated on synthetic Gaussian experiment
- CIFAR-10 DDPM pipeline operational
- Two discriminator targets demonstrated:
  - Class guidance (airplane vs others)
  - Quality guidance (sharp vs blurry)

## Files

| File | Description |
|---|---|
| `DSGtest.ipynb` | Unguided DDPM and DSG implementation |
| `THMproof.ipnby` | Validation of theorem |

## Team
[J Ashmita] · [Zoya Bothra] · [Amithi.S] · [Srijan Chopra]

