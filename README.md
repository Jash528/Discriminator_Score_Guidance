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
| [DSGtest.py](DSGtest.py) | Unguided DDPM and DSG implementation |
| [THMproof.py](THMproof.py) | Validation of theorem |
| [GAN_final.py](GAN_final.py) | Final DSG implementation |
| [DSG.py](DSG.py) | Continuation of final DSG implementation |
| [Project Website](https://discriminator-score-guidence.netlify.app/) | Link to the DSG project site |


## Team
[J Ashmita] · [Zoya Bothra] · [Amithi.S] · [Srijan Chopra]

