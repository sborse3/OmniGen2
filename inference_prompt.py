import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLModel as TextEncoder
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.transport import create_transport
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from torchvision.transforms.functional import to_pil_image
import os


class PrefixTokens(torch.nn.Module):
    """Trainable prefix tokens that are prepended to text encoder input."""
    
    def __init__(self, num_prefix_tokens, hidden_size, dtype=torch.float32):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_size = hidden_size
        self.prefix_embeddings = torch.nn.Parameter(
            torch.randn(1, num_prefix_tokens, hidden_size, dtype=dtype) * 0.02
        )
    
    def forward(self, batch_size):
        """Return prefix embeddings for the given batch size."""
        return self.prefix_embeddings.expand(batch_size, -1, -1)


class PrefixTextEncoder(torch.nn.Module):
    """Wrapper around text encoder that adds trainable prefix tokens to the output."""
    def __init__(self, text_encoder, num_prefix_tokens, dtype=torch.float32):
        super().__init__()
        self.text_encoder = text_encoder
        self.prefix_tokens = PrefixTokens(num_prefix_tokens, text_encoder.config.hidden_size, dtype)
        
    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        # Run the text encoder as usual
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )
        text_hidden_states = encoder_outputs.last_hidden_state
        
        # Add prefix tokens to the output
        prefix_embeddings = self.prefix_tokens(batch_size)
        
        # Ensure both tensors are on the same device and have the same dtype
        prefix_embeddings = prefix_embeddings.to(device=text_hidden_states.device, dtype=text_hidden_states.dtype)
        
        # Concatenate prefix tokens at the beginning of the sequence
        combined_hidden_states = torch.cat([prefix_embeddings, text_hidden_states], dim=1)
        
        # Ensure the concatenated tensor is contiguous
        combined_hidden_states = combined_hidden_states.contiguous()
        
        # If output_hidden_states is requested, append to the list
        hidden_states_list = [combined_hidden_states] if output_hidden_states else None
        return type('TextEncoderOutput', (), {
            'last_hidden_state': combined_hidden_states,
            'hidden_states': hidden_states_list
        })()


def load_trained_prefix_tokens(checkpoint_path, num_prefix_tokens, hidden_size, device, dtype):
    """Load trained prefix tokens from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Find prefix token parameters in the checkpoint
    prefix_state_dict = {}
    for key, value in checkpoint.items():
        if 'prefix_tokens' in key:
            prefix_state_dict[key] = value
    
    if not prefix_state_dict:
        raise ValueError("No prefix token parameters found in checkpoint")
    
    # Create prefix tokens module and load state
    prefix_tokens = PrefixTokens(num_prefix_tokens, hidden_size, dtype)
    prefix_tokens.load_state_dict(prefix_state_dict)
    prefix_tokens = prefix_tokens.to(device)
    
    return prefix_tokens


def main():
    parser = argparse.ArgumentParser(description="Inference with trained prefix tokens")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to training config")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape", help="Text prompt")
    parser.add_argument("--output_path", type=str, default="output.png", help="Output image path")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.train.mixed_precision == "bf16" else torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Load models
    print("Loading models...")
    
    # Load text encoder
    text_encoder = TextEncoder.from_pretrained(
        config.model.pretrained_text_encoder_model_name_or_path,
        torch_dtype=dtype,
    )
    
    # Load trained prefix tokens
    num_prefix_tokens = config.train.get('num_prefix_tokens', 10)
    prefix_tokens = load_trained_prefix_tokens(
        args.checkpoint_path, 
        num_prefix_tokens, 
        text_encoder.config.hidden_size, 
        device, 
        dtype
    )
    
    # Wrap text encoder with prefix tokens
    text_encoder = PrefixTextEncoder(text_encoder, num_prefix_tokens, dtype)
    text_encoder.prefix_tokens = prefix_tokens  # Replace with trained prefix tokens
    text_encoder = text_encoder.to(device)
    
    # Load main model
    model = OmniGen2Transformer2DModel.from_pretrained(
        config.model.pretrained_model_path, subfolder="transformer"
    )
    model = model.to(device, dtype=dtype)
    model.eval()
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config.model.pretrained_vae_model_name_or_path,
        subfolder=config.model.get("vae_subfolder", "vae"),
    )
    vae = vae.to(device, dtype=dtype)
    vae.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_text_encoder_model_name_or_path)
    
    # Create scheduler
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        config.model.pretrained_model_path, subfolder="scheduler"
    )
    
    # Create transport
    transport = create_transport(
        "Linear",
        "velocity",
        None,
        None,
        None,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=config.data.max_output_pixels // 16 // 16,
        dynamic_time_shift=config.transport.get("dynamic_time_shift", False),
        time_shift_version=config.transport.get("time_shift_version", "v1"),
    )
    
    # Get rotary embeddings
    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
        model.config.axes_dim_rope,
        model.config.axes_lens,
        theta=10000,
    )
    
    print("Models loaded successfully!")
    
    # Tokenize prompt
    print(f"Processing prompt: {args.prompt}")
    text_inputs = tokenizer(
        [args.prompt],
        padding="longest",
        max_length=config.data.maximum_text_tokens,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    text_mask = text_inputs.attention_mask.to(device)
    
    # Encode text
    with torch.no_grad():
        text_feats = text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_mask,
            output_hidden_states=False,
        ).last_hidden_state

    # Update text_mask to match new sequence length (prefix + text)
    batch_size = text_mask.shape[0]
    num_prefix_tokens = config.train.get('num_prefix_tokens', 10)
    prefix_mask = torch.ones(batch_size, num_prefix_tokens, device=text_mask.device, dtype=text_mask.dtype)
    text_mask_with_prefix = torch.cat([prefix_mask, text_mask], dim=1)

    # Generate rotary embeddings for the new sequence length (prefix + text)
    total_seq_len = text_feats.shape[1]  # This now includes prefix tokens
    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
        model.config.axes_dim_rope,
        [total_seq_len] * len(model.config.axes_dim_rope),
        theta=10000,
    )
    
    # Initialize latents
    batch_size = 1
    height = width = int(math.sqrt(config.data.max_output_pixels))
    latents = torch.randn(
        (batch_size, 16, height // 8, width // 8),
        device=device,
        dtype=dtype
    )
    
    # Set timesteps
    scheduler.set_timesteps(args.num_inference_steps)
    timesteps = scheduler.timesteps
    
    print(f"Starting inference with {args.num_inference_steps} steps...")
    
    # Denoising loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Prepare model inputs
            model_kwargs = dict(
                text_hidden_states=text_feats,
                text_attention_mask=text_mask_with_prefix,
                ref_image_hidden_states=None,  # No reference images for now
                freqs_cis=freqs_cis,
            )
            
            # Predict noise
            noise_pred = model(
                latents,
                t,
                **model_kwargs
            )
            
            # Apply classifier-free guidance
            if args.guidance_scale > 1.0:
                # Unconditional prediction
                uncond_model_kwargs = dict(
                    text_hidden_states=torch.zeros_like(text_feats),
                    text_attention_mask=text_mask_with_prefix,
                    ref_image_hidden_states=None,
                    freqs_cis=freqs_cis,
                )
                
                uncond_noise_pred = model(
                    latents,
                    t,
                    **uncond_model_kwargs
                )
                
                noise_pred = uncond_noise_pred + args.guidance_scale * (noise_pred - uncond_noise_pred)
            
            # Step with scheduler
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            if (i + 1) % 5 == 0:
                print(f"Step {i+1}/{len(timesteps)}")
    
    # Decode latents
    print("Decoding latents...")
    if vae.config.scaling_factor is not None:
        latents = latents / vae.config.scaling_factor
    if vae.config.shift_factor is not None:
        latents = latents + vae.config.shift_factor
    
    image = vae.decode(latents, return_dict=False)[0]
    image = image.clamp(-1, 1)
    
    # Save image
    image_pil = to_pil_image(image[0] * 0.5 + 0.5)
    image_pil.save(args.output_path)
    
    print(f"Image saved to {args.output_path}")


if __name__ == "__main__":
    import math
    main()
