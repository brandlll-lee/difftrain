# difftrain MVP Spec (Text-to-Image, SD-style UNet)

Status: Draft (design-first, no implementation)

This document defines the minimal, stable interfaces for the first runnable
difftrain MVP. It is intentionally narrow to avoid churn. All future features
must extend these interfaces rather than replace them.

## Scope (MVP)

- Modality: text-to-image
- Training space: latent diffusion (VAE-encoded latents)
- Model family: SD-style conditional UNet
- Text conditioning: CLIP text encoder
- Schedulers: DDPM-style (alphas_cumprod + add_noise)
- Training strategies: DREAM (epsilon + v_prediction)

Non-goals for MVP:
- pixel-space training
- video/audio/3D modalities
- DiT/Transformer-based diffusion backbones
- multiple text encoders or multimodal conditioning

## Terminology

- x0: clean latent
- xt: noisy latent at timestep t
- eps: noise
- alpha_t, sigma_t: sqrt(alphas_cumprod), sqrt(1 - alphas_cumprod)
- prediction_type: "epsilon" | "v_prediction" | "sample"

## Batch Contract (Text-to-Image)

Training batch MUST expose the following fields (directly or via a dict):

- image: raw image tensor or path (pre-VAE)
- input_ids: token ids for CLIP text encoder
- latents: VAE-encoded latents (optional precompute)
- timesteps: sampled timesteps (per-sample)
- metadata: optional misc info

Notes:
- If latents are not provided, the training loop must call the VAE.
- input_ids are required for CLIP conditioning in MVP.

## Model Interface Contract

The UNet model MUST expose:

model(noisy_latents, timesteps, encoder_hidden_states) -> prediction

Where:
- noisy_latents: xt
- timesteps: tensor of shape [B]
- encoder_hidden_states: CLIP text embeddings
- prediction semantics are controlled by prediction_type.

## Scheduler Interface Contract

Scheduler MUST provide:

- alphas_cumprod: 1D tensor of length num_train_timesteps
- add_noise(x0, noise, timesteps) -> xt
- get_velocity(x0, noise, timesteps) -> v (for v_prediction)
- prediction_type: config field ("epsilon"|"v_prediction"|"sample")

## Target Construction

Given x0, noise, timesteps:

- If prediction_type == "epsilon":
  target = noise
- If prediction_type == "v_prediction":
  target = scheduler.get_velocity(x0, noise, timesteps)
- If prediction_type == "sample":
  target = xt (noisy latents)

## DREAM Training Strategy (MVP)

DREAM adjusts xt and target using one extra forward pass without gradients.
This is implemented as an optional strategy hook.

Core definitions:

alpha_t = sqrt(alphas_cumprod[t])
sigma_t = sqrt(1 - alphas_cumprod[t])
dream_lambda = sigma_t ** p

### epsilon prediction

pred_eps = model(xt, t, cond)
delta_eps = (eps - pred_eps).detach()

xt' = xt + sigma_t * (delta_eps * dream_lambda)
target' = target + (delta_eps * dream_lambda)

### v_prediction

pred_v = model(xt, t, cond)
pred_eps = sigma_t * xt + alpha_t * pred_v
delta_eps = (eps - pred_eps).detach()

xt' = xt + sigma_t * (delta_eps * dream_lambda)
target' = target + alpha_t * (delta_eps * dream_lambda)

Notes:
- This keeps x_t consistent with updated epsilon while preserving x0.
- For v_prediction, target is v, so the correction uses alpha_t.

## Training Loop (Pseudo)

1. Encode image -> latents (VAE) if not precomputed
2. Sample timesteps
3. Sample noise
4. xt = scheduler.add_noise(latents, noise, timesteps)
5. encoder_hidden_states = clip_text_encoder(input_ids)
6. Build target by prediction_type
7. If dream_training:
   xt, target = dream_update(...)
8. pred = model(xt, timesteps, encoder_hidden_states)
9. loss = mse(pred, target)
10. backward + optimizer step

## Configuration Fields (MVP)

train:
- dream_training: bool
- dream_detail_preservation: float (p)
- precision: "fp32" | "fp16" | "bf16"
- resume_from: str (checkpoint path or empty)

model:
- prediction_type: "epsilon" | "v_prediction" | "sample"

## Extension Points (Post-MVP)

- New modalities: add new batch contract + encoder modules
- New models: swap UNet with DiT/Transformer
- New training space: pixel-space (bypass VAE)
- New text encoders: T5/SigLIP/Qwen-VL
- New strategies: Min-SNR, EMA, P2, LCM distill, etc

All extensions must preserve the above contracts or provide explicit adapters.
