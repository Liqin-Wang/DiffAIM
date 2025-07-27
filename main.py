import os
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import lpips

from config import get_args
from diffusers import StableDiffusionPipeline, DDIMScheduler
from utils.attack_utils import get_fr_model, compute_weighted_score
from utils.dataset import ImagePairDataset
from utils.inversion_utils import encode_image, encode_text, ddpm_inversion, decode_image, inversion_reverse_one_step
from utils.attentionControl import AttentionStore, register_attention_control


def main():
    args = get_args()
    device = f"cuda:{args.device_id}"

    # Load models
    ldm = StableDiffusionPipeline.from_pretrained(args.ldm_id).to(device)
    ldm.scheduler = DDIMScheduler.from_config(args.ldm_id, subfolder="scheduler")
    ldm.scheduler.set_timesteps(args.num_diffusion_steps)
    ldm.unet.sample_size = 32 if args.image_size == 256 else 64  # adjust for 256x256 or 512x512 images

    fr_model_dict = {
        'IR152': get_fr_model('IR152', device),
        'IRSE50': get_fr_model('IRSE50', device),
        'FaceNet': get_fr_model('FaceNet', device),
        'MobileFace': get_fr_model('MobileFace', device)
    }
    target_model_name = args.target_model
    surrogate_models = {model_name: model for model_name, model in fr_model_dict.items() if model_name != target_model_name}

    LPIPSloss = lpips.LPIPS(net='alex').to(device)

    # Prepare dataset
    dataset = ImagePairDataset(data_dir=args.dataset, target_size=args.image_size)
    dataset = Subset(dataset, [x for x in range(args.num)])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    save_path = f'{args.save}/{target_model_name}'
    os.makedirs(save_path, exist_ok=True)

    # Process attack
    print(f"Processing attack model: {target_model_name}")
    for i, (src_img, tgt_img, src_name, tgt_name) in enumerate(dataloader):
        src_img = src_img.to(device)
        tgt_img = tgt_img.to(device)

        print(f"Processing image {i+1}/{len(dataloader)}: {src_name[0]}")

        # DDPM inversion
        x0 = encode_image(ldm, src_img)
        xt, zs_all, xts_all = ddpm_inversion(ldm, x0, args.eta, cfg_scale=args.cfg_src, prog_bar=True,
                                             num_inference_steps=args.num_diffusion_steps)
        xT = xts_all[args.num_diffusion_steps - args.skip]
        xTs = xts_all[:(args.num_diffusion_steps - args.skip) + 1]
        zs = zs_all[:(args.num_diffusion_steps - args.skip)]

        timesteps = ldm.scheduler.timesteps.to(device)
        xt = xT.unsqueeze(0) if xT.dim() == 3 else xT
        op = tqdm(timesteps[-zs.shape[0]:])
        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

        # Initialize ensemble weights
        weights_list = [1 / len(surrogate_models)] * len(surrogate_models)
        weights = torch.tensor(weights_list, device=device)

        # Setup attention controller
        controller = AttentionStore()
        register_attention_control(ldm, controller)

        text_embedding = None
        uncond_embedding = encode_text(ldm, "")

        for _, t in enumerate(op):
            idx = (ldm.scheduler.num_inference_steps - t_to_idx[int(t)] - (ldm.scheduler.num_inference_steps - zs.shape[0] + 1))

            xt_benign = xTs[idx + 1].unsqueeze(0) if xTs[idx + 1].dim() == 3 else xTs[idx + 1]
            xt_benign.requires_grad_(False)

            xt_ori = xt.detach().clone()
            out = inversion_reverse_one_step(ldm, xt_ori, t, zs[idx], uncond_embedding, text_embedding, args.cfg_tar, args.eta,
                                             controller)
            xtm1_ori = out["xtm1"].detach()

            for _ in range(args.steps):
                controller.reset()
                controller.loss = 0

                xt_adv = xt.detach().clone().requires_grad_(True)
                xt_latents = torch.cat([xt_benign, xt_adv])
                out = inversion_reverse_one_step(ldm, xt_latents, t, zs[idx], uncond_embedding, text_embedding, args.cfg_tar,
                                                 args.eta, controller, requires_grad=True)
                pred_x0 = out["pred_x0"]
                img = ldm.vae.decode(1 / 0.18215 * pred_x0).sample

                # Identity Convergence
                id_loss, weights = compute_weighted_score(surrogate_models, img, src_img, tgt_img, weights, diversity=5)

                # Semantic Divergence
                middle_feature = controller.middle_feature
                middle_feature = middle_feature.reshape(middle_feature.shape[0], middle_feature.shape[1], -1)
                semantic_loss = 1 - F.cosine_similarity(middle_feature[0], middle_feature[1]).mean()

                #  Structural Preservation
                structural_loss = controller.loss

                # LPIPS Loss
                lpips_loss = LPIPSloss(img, src_img)
                lpips_loss = max(torch.tensor(0), lpips_loss - args.lpips_eps)

                op.set_postfix(
                    id_score=id_loss.item(),
                    semantic_loss=semantic_loss.item(),
                    structural_loss=structural_loss.item(),
                    lpips_loss=lpips_loss.item(),
                )

                loss = 10 + id_loss * args.id_loss_weight \
                    + semantic_loss * args.semantic_loss_weight \
                    - structural_loss * args.structural_loss_weight \
                        - lpips_loss * args.lpips_loss_weight

                xt_adv.grad = torch.autograd.grad(loss, xt_adv)[0]

                xt_adv = xt_adv + xt_adv.grad * args.step_size
                noise = (xt_adv - xt_ori).clamp(-args.eps, args.eps)
                xt_adv = xt_ori + noise
                xt = xt_adv.detach()

            xt = xtm1_ori + (xt_adv - xt_ori)

        # Generate final adversarial image
        x0_adv = xt.detach().clone()
        x_adv = decode_image(ldm, x0_adv)
        save_image((x_adv + 1) / 2, f"{save_path}/{src_name[0].split('.')[0]}.png")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
