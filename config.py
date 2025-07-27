import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Device and model settings
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--ldm_id", type=str, default="./pretrained/stable-diffusion-v1-5",
                        help="Path to the diffusion model, stable-diffusion-v1-4 is also available")
    parser.add_argument("--cfg_src", type=float, default=3.5, help="CFG scale for source")
    parser.add_argument("--cfg_tar", type=float, default=15, help="CFG scale for target")
    parser.add_argument("--num_diffusion_steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--eta", type=float, default=1, help="DDIM eta parameter")

    # Dataset settings
    parser.add_argument("--dataset", default="./dataset", help="Dataset directory path")
    parser.add_argument("--image_size", type=int, default=256, help="Image size, 256 or 512")
    parser.add_argument("--num", type=int, default=5, help="Number of images to process")

    # Attack settings
    parser.add_argument("--target_model", type=str, default="IR152", help="Target fr model name",
                        choices=["IR152", "IRSE50", "FaceNet", "MobileFace"])
    parser.add_argument("--skip", type=int, default=80, help="Skip steps")
    parser.add_argument("--eps", type=float, default=0.05, help="Perturbation epsilon")
    parser.add_argument("--steps", type=int, default=10, help="Attack steps")
    parser.add_argument("--step_size", type=float, default=3, help="Step size for adversarial optimization")

    # Loss weights
    parser.add_argument("--id_loss_weight", type=float, default=1, help="Identity convergence weight")
    parser.add_argument("--semantic_loss_weight", type=float, default=2, help="Semantic divergence weight")
    parser.add_argument("--structural_loss_weight", type=float, default=0.1, help="Structural preservation weight")
    parser.add_argument("--lpips_loss_weight", type=float, default=8, help="Lpips loss weight")
    parser.add_argument("--lpips_eps", type=float, default=0.01, help="Lpips loss epsilon")

    # Output settings
    parser.add_argument("--save", default="results", help="Image save path")

    return parser.parse_args()
