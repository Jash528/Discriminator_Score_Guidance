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
| [DSGtest.ipynb](DSGtest.ipynb) | Unguided DDPM and DSG implementation |
| [THMproof.ipynb](THMproof.ipynb) | Validation of theorem |
| [GAN_final.ipynb](GAN_final.ipynb) | Final DSG implementation |
| [DSG.ipynb](DSG.ipynb) | Continuation of final DSG implementation |
| [Project Website](https://discriminator-score-guidence.netlify.app/) | Link to the DSG project site |


## Team
[J Ashmita] · [Zoya Bothra] · [Amithi.S] · [Srijan Chopra]

