import torch
from tqdm import tqdm
from torch import autocast, inference_mode


def encode_image(model, image):
    with autocast("cuda"), inference_mode():
        image_encoding = (model.vae.encode(image).latent_dist.mode() * 0.18215).float()
    return image_encoding


def decode_image(model, image_encoding):
    with autocast("cuda"), inference_mode():
        image = model.vae.decode(1 / 0.18215 * image_encoding).sample
    return image


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    '''Samples from P(x_1:T|x_0)

    Return:
      xts: [x_0, x_1, x_2, ..., x_T]
    '''
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar)**0.5

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(
        (num_inference_steps + 1, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t]**0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]

    return xts


def get_noise_pred(model, xt, t, uncond_embedding, cond_embedding=None, cfg_scale=3.5, requires_grad=False):
    with torch.set_grad_enabled(requires_grad):
        out = model.unet(xt, timestep=t, encoder_hidden_states=uncond_embedding)
        if cond_embedding is not None:
            cond_out = model.unet(xt, timestep=t, encoder_hidden_states=cond_embedding)
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample
    return noise_pred


def get_mu_xt(model, xt, t, noise_pred, eta):
    '''
    Return:
        mu_xt: 
        pred_original_sample: P(ft(xt)), predicted x0
        pred_sample_direction: D(ft(xt)), direction pointing to xt
        variance: variance schedule at timestep t
    '''
    alpha_bar = model.scheduler.alphas_cumprod
    prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_bar[t]
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_bar[t] / alpha_prod_t_prev)

    # pred of x0，P(ft(xt))
    pred_original_sample = (xt - (1 - alpha_bar[t])**0.5 * noise_pred) / alpha_bar[t]**0.5

    # direction to xt，D(ft(xt))
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance)**(0.5) * noise_pred

    mu_xt = alpha_prod_t_prev**(0.5) * pred_original_sample + pred_sample_direction

    return mu_xt, pred_original_sample, pred_sample_direction, variance


def ddpm_inversion(model, x0, eta, prog_bar=False, prompt="", cfg_scale=3.5, num_inference_steps=50):
    '''
    Return:
        xt: = xts[1]
        zs: [0, z_1, z_2, ..., z_T]
        xts: [x_0, x_1, x_2, ..., x_T]
    '''
    text_embedding = encode_text(model, prompt) if not prompt == "" else None
    uncond_embedding = encode_text(model, "")

    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (num_inference_steps, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)
    zs = torch.zeros(size=variance_noise_shape, device=model.device)

    xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps
    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1

        xt = xts[idx + 1][None]  # because xts[0] = x0
        noise_pred = get_noise_pred(model, xt, t, uncond_embedding, text_embedding, cfg_scale)
        mu_xt, _, _, variance = get_mu_xt(model, xt, t, noise_pred, eta)

        xtm1 = xts[idx][None]  # x_{t-1}
        z = (xtm1 - mu_xt) / (eta * variance**0.5)
        zs[idx] = z

        # correction to avoid error accumulation
        xtm1 = mu_xt + (eta * variance**0.5) * z
        xts[idx] = xtm1

    zs[0] = torch.zeros_like(zs[0])
    return xt, zs, xts


def inversion_reverse_one_step(model, xt, t, z, uncond_embedding, text_embedding, cfg_scale, eta, controller=None,
                               requires_grad=False) -> dict:
    '''Predict x_{t-1} from x_t with noise z
    
    Return:
        mu_xt: 
        pred_x0: P(f_t(x_t)), predicted x0
        pred_sample_direction: D(f_t(x_t)), direction pointing to x_t
        variance: variance schedule at timestep t
        sigma_z: eta * variance**(0.5) * z
        xtm1: x_{t-1}
    '''
    noise_pred = get_noise_pred(model, xt, t, uncond_embedding, text_embedding, cfg_scale, requires_grad)
    if noise_pred.shape[0] == 2:
        noise_pred = noise_pred[1][None]
        xt = xt[1][None]
    variance_noise = z.unsqueeze(0) if z.dim() == 3 else z
    mu_xt, pred_x0, pred_sample_direction, variance = get_mu_xt(model, xt, t, noise_pred, eta)
    sigma_z = eta * variance**(0.5) * variance_noise
    xtm1 = mu_xt + sigma_z

    if controller is not None:
        xtm1 = controller.step_callback(xtm1)

    out = {
        "mu_xt": mu_xt,
        "pred_x0": pred_x0,
        "pred_sample_direction": pred_sample_direction,
        "variance": variance,
        "sigma_z": sigma_z,
        "xtm1": xtm1
    }
    return out


def inversion_reverse_process(model, xT, eta=0, prompt="", cfg_scale=15, prog_bar=False, zs=None, controller=None, **kwargs):

    text_embedding = encode_text(model, prompt) if not prompt == "" else None
    uncond_embedding = encode_text(model, "")

    timesteps = model.scheduler.timesteps.to(model.device)
    xt = xT.unsqueeze(0) if xT.dim() == 3 else xT
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = model.scheduler.num_inference_steps - t_to_idx[int(t)] - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        out = inversion_reverse_one_step(model, xt, t, zs[idx], uncond_embedding, text_embedding, cfg_scale, eta, controller)
        xt = out["xtm1"]
    return xt
