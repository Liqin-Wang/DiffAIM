import torch
import abc
from typing import Dict, List

LOW_RESOURCE = True


def register_attention_control(model, controller):
    """
    Register attention control hooks to the diffusion model.
    """

    def create_attention_forward(attention_module, place_in_unet: str):
        to_out = _get_output_layer(attention_module)

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            context = encoder_hidden_states
            is_cross_attention = context is not None
            batch_split_idx = hidden_states.shape[0] // 2

            has_multi_input = batch_split_idx > 0

            sim_original = None
            if has_multi_input and not is_cross_attention:
                sim_original = _compute_attention_similarity(attention_module, hidden_states[:batch_split_idx], context,
                                                             is_cross_attention)

            adversarial_input = hidden_states[batch_split_idx:] if has_multi_input else hidden_states
            sim_adversarial = _compute_attention_similarity(attention_module, adversarial_input, context, is_cross_attention)

            # Calculate attention loss for self-attention layers
            if has_multi_input and not is_cross_attention and sim_original is not None:
                _accumulate_attention_loss(controller, sim_original, sim_adversarial)

            # Apply attention and return output
            attention_weights = sim_adversarial.softmax(dim=-1)
            attention_weights = controller(attention_weights, is_cross_attention, place_in_unet)

            return _apply_attention_output(attention_module, attention_weights, adversarial_input, context, to_out)

        return forward

    def _get_output_layer(attention_module):
        """Get the output layer from attention module."""
        to_out = attention_module.to_out
        if isinstance(to_out, torch.nn.modules.container.ModuleList):
            return to_out[0]
        return to_out

    def _compute_attention_similarity(attention_module, hidden_states, context, is_cross_attention):
        """Compute attention similarity matrix."""
        q = attention_module.to_q(hidden_states)
        context = context if is_cross_attention else hidden_states
        k = attention_module.to_k(context)

        q = attention_module.head_to_batch_dim(q)
        k = attention_module.head_to_batch_dim(k)

        return torch.einsum("b i d, b j d -> b i j", q, k) * attention_module.scale

    def _accumulate_attention_loss(controller, sim_original, sim_adversarial):
        if sim_adversarial.shape[1] <= 32**2:  # avoid memory overhead
            controller.loss += controller.criterion(sim_original, sim_adversarial)

    def _apply_attention_output(attention_module, attention_weights, hidden_states, context, to_out):
        context = context if context is not None else hidden_states
        v = attention_module.to_v(context)
        v = attention_module.head_to_batch_dim(v)

        out = torch.einsum("b i j, b j d -> b i d", attention_weights, v)
        out = attention_module.batch_to_head_dim(out)
        return to_out(out)

    def register_recursive(network, count: int, place_in_unet: str) -> int:
        if network.__class__.__name__ == 'Attention':
            network.forward = create_attention_forward(network, place_in_unet)
            return count + 1
        elif hasattr(network, 'children'):
            for child in network.children():
                count = register_recursive(child, count, place_in_unet)
        return count

    def create_feature_hook(controller):
        """
        Create a hook to save features from the middle of the UNet.
        """

        def save_features(module, input_data, output_data, key):
            if output_data[0].shape[0] == 2:
                controller.middle_feature = output_data[0]

        return save_features

    def register_downblock_hooks(network):
        if network.__class__.__name__ == 'DownBlock2D':
            hook_fn = create_feature_hook(controller)
            network.register_forward_hook(lambda m, i, o, name=network.__class__.__name__: hook_fn(m, i, o, name))

    # Register attention control on all UNet components
    attention_layer_count = 0
    for name, subnet in model.unet.named_children():
        if "down" in name:
            attention_layer_count += register_recursive(subnet, 0, "down")
        elif "up" in name:
            attention_layer_count += register_recursive(subnet, 0, "up")
        elif "mid" in name:
            attention_layer_count += register_recursive(subnet, 0, "mid")

        # Register feature extraction hooks for down blocks
        if name == 'down_blocks':
            register_downblock_hooks(subnet[-1])

    controller.num_att_layers = attention_layer_count


class AttentionControl(abc.ABC):

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        pass

    @property
    def num_uncond_att_layers(self) -> int:
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                half_batch = attn.shape[0] // 2
                attn[half_batch:] = self.forward(attn[half_batch:], is_cross, place_in_unet)

        self._advance_layer_counter()
        return attn

    def _advance_layer_counter(self):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    """
    Attention storage controller that collects and stores attention maps.
    """

    def __init__(self):
        super().__init__()
        self.step_store = self._get_empty_store()
        self.attention_store = {}
        self.loss = 0
        self.criterion = torch.nn.MSELoss()
        self.middle_feature = None

    @staticmethod
    def _get_empty_store() -> Dict[str, List]:
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if not self.attention_store:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]

        self.step_store = self._get_empty_store()

    def get_average_attention(self) -> Dict[str, List]:
        if self.cur_step == 0:
            return self.attention_store

        return {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}

    def reset(self):
        super().reset()
        self.step_store = self._get_empty_store()
        self.attention_store = {}
        self.middle_feature = None
        self.loss = 0
