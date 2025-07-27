import torch
import torch.nn.functional as F
from models import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1


def input_noise(x, diversity_prob=0.5, noise_scale=0.1):
    """
    Add random noise to input tensor for data augmentation.
    
    Args:
        x: Input tensor
        diversity_prob: Probability of applying noise
        noise_scale: Scale of the noise to be added
    
    Returns:
        Noised tensor or original tensor based on probability
    """
    if torch.rand(1) < diversity_prob:
        rnd = torch.rand(1, device=x.device)
        noise = torch.randn_like(x, device=x.device)
        return x + rnd * (noise_scale**0.5) * noise
    return x


def input_diversity(x, resize_rate=0.9, diversity_prob=0.5):
    """
    Apply input diversity transformation including random resizing and padding.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        resize_rate: Rate for resizing the image
        diversity_prob: Probability of applying diversity transformation
    
    Returns:
        Transformed tensor or original tensor based on probability
    """
    if torch.rand(1) >= diversity_prob:
        return x

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size, img_resize = img_resize, img_size

    # Random resize
    rnd_size = torch.randint(low=img_size, high=img_resize + 1, size=(1, ), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd_size, rnd_size], mode='bilinear', align_corners=False)

    # Random padding
    h_rem = w_rem = img_resize - rnd_size
    pad_top = torch.randint(low=0, high=h_rem.item() + 1, size=(1, ), dtype=torch.int32)
    pad_left = torch.randint(low=0, high=w_rem.item() + 1, size=(1, ), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    return padded


def compute_weighted_score(surrogate_models, img, x_src, x_target, weights, diversity=5):
    """
    Compute weighted similarity score across multiple fr models with input diversity.
    
    Args:
        surrogate_models: Dictionary of fr models
        img: Current adversarial image
        x_src: Source image
        x_target: Target image
        weights: Current weights for each classifier
        diversity: Number of diverse inputs to generate
    
    Returns:
        Tuple of (weighted_score, new_weight_tensor)
    """
    batch_size = img.shape[0]
    score_list = []

    # Generate diverse inputs
    diverse_inputs = [input_diversity(input_noise(img)) for _ in range(diversity)]

    for model_name, model in surrogate_models.items():
        input_size = (160, 160) if model_name == 'FaceNet' else (112, 112)
        resize_layer = torch.nn.AdaptiveAvgPool2d(input_size)

        target_features = model(resize_layer(x_target)).reshape(batch_size, -1)
        source_features = model(resize_layer(x_src)).reshape(batch_size, -1)

        # Compute average score across diverse inputs
        total_score = 0
        for diverse_img in diverse_inputs:
            current_features = model(resize_layer(diverse_img)).reshape(batch_size, -1)

            target_sim = F.cosine_similarity(current_features, target_features).sum() / batch_size
            source_sim = F.cosine_similarity(current_features, source_features).sum() / batch_size
            score = target_sim - source_sim + 1
            total_score += score

        avg_score = total_score / diversity
        score_list.append(avg_score)

    # Compute weighted score and adaptive weights
    score_tensor = torch.stack(score_list)
    weighted_score = torch.sum(score_tensor * weights)
    new_weights = torch.exp(1 - score_tensor) / torch.sum(torch.exp(1 - score_tensor))

    return weighted_score, new_weights


def get_fr_model(name: str, device: torch.device):
    """
    Load and initialize face recognition model.
    """
    model_configs = {
        'IRSE50': (IRSE_50, 'pretrained/irse50.pth'),
        'MobileFace': (lambda: MobileFaceNet(512), 'pretrained/mobile_face.pth'),
        'IR152': (lambda: IR_152([112, 112]), 'pretrained/ir152.pth'),
        'FaceNet': (lambda: InceptionResnetV1(num_classes=8631), 'pretrained/facenet.pth')
    }

    if name not in model_configs:
        raise ValueError(f'Unsupported model name: {name}. Supported models: {list(model_configs.keys())}')

    model_class, pretrained_path = model_configs[name]
    model = model_class()

    try:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    except FileNotFoundError:
        raise FileNotFoundError(f'Pretrained model file not found: {pretrained_path}')

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)


def asr_calculation(sim_scores_dict):
    """
    Calculate Attack Success Rate (ASR) for different False Accept Rate (FAR) thresholds.
    
    Args:
        sim_scores_dict: Dictionary mapping model names to lists of similarity scores
    
    Returns:
        Dictionary containing ASR results for each model and threshold
    """
    threshold_dict = {
        'IR152': {
            'far_0.1': 0.094632,
            'far_0.01': 0.166788,
            'far_0.001': 0.227922
        },
        'IRSE50': {
            'far_0.1': 0.144840,
            'far_0.01': 0.241045,
            'far_0.001': 0.312703
        },
        'FaceNet': {
            'far_0.1': 0.256587,
            'far_0.01': 0.409131,
            'far_0.001': 0.591191
        },
        'MobileFace': {
            'far_0.1': 0.183635,
            'far_0.01': 0.301611,
            'far_0.001': 0.380878
        }
    }

    results = {}

    for model_name, scores in sim_scores_dict.items():
        if model_name not in threshold_dict:
            print(f"Warning: No thresholds defined for model {model_name}")
            continue

        model_results = {}
        total_samples = len(scores)

        if total_samples == 0:
            print(f"Warning: No scores found for model {model_name}")
            continue

        thresholds = threshold_dict[model_name]

        for threshold_name, threshold_value in thresholds.items():
            successful_attacks = sum(1 for score in scores if score > threshold_value)
            asr = round(successful_attacks * 100 / total_samples, 5)
            model_results[threshold_name] = asr

            print(f"{model_name} attack success rate ({threshold_name}): {asr}%")

        results[model_name] = model_results

    return results
