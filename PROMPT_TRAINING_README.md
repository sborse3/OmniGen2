# OmniGen2 Prompt Training

This directory contains the implementation for training OmniGen2 with trainable prefix tokens instead of LoRA. The prefix tokens are prepended to the text encoder input and are optimized during training to improve the model's performance on specific tasks.

## Overview

The prompt training approach adds a small number of trainable prefix tokens to the text encoder input. These tokens are learned during training and can help the model better understand and generate images for specific domains or styles.

### Key Features

- **Prefix Tokens**: Configurable number of trainable prefix tokens (default: 10)
- **Efficient Training**: Only the prefix tokens are trained, keeping the main model frozen
- **Compatible**: Uses the same training infrastructure as LoRA training
- **Memory Efficient**: Much smaller memory footprint compared to full fine-tuning

## Files

- `train_prompt.py`: Main training script for prefix token training
- `inference_prompt.py`: Inference script to test trained prefix tokens
- `options/ft_prompt.yml`: Configuration file for prompt training
- `scripts/train/ft_prompt.sh`: Training script wrapper

## Usage

### Training

1. **Prepare your data**: Ensure your data is in the correct format as specified in the OmniGen2 documentation.

2. **Configure training**: Edit `options/ft_prompt.yml` to set your desired parameters:
   ```yaml
   train:
     num_prefix_tokens: 10  # Number of prefix tokens to train
     learning_rate: 1e-4    # Learning rate for prefix tokens
     max_train_steps: 4000  # Number of training steps
   ```

3. **Start training**:
   ```bash
   ./scripts/train/ft_prompt.sh
   ```

   Or run directly:
   ```bash
   accelerate launch train_prompt.py --config options/ft_prompt.yml
   ```

### Inference

After training, you can use the trained prefix tokens for inference:

```bash
python inference_prompt.py \
    --checkpoint_path experiments/ft_prompt/checkpoint-4000/pytorch_model.bin \
    --config_path options/ft_prompt.yml \
    --prompt "A beautiful landscape with mountains" \
    --output_path output.png \
    --num_inference_steps 20 \
    --guidance_scale 7.5
```

## Configuration

### Key Parameters

- `num_prefix_tokens`: Number of trainable prefix tokens (default: 10)
- `learning_rate`: Learning rate for prefix tokens (default: 1e-4)
- `max_train_steps`: Number of training steps (default: 4000)
- `batch_size`: Batch size per device (default: 2)
- `global_batch_size`: Total batch size across all devices (default: 16)

### Comparison with LoRA

| Aspect | LoRA Training | Prompt Training |
|--------|---------------|-----------------|
| Trainable Parameters | ~1-2M | ~20K (10 tokens × 2048 dim) |
| Memory Usage | Medium | Low |
| Training Speed | Medium | Fast |
| Flexibility | High | Medium |
| Storage Size | Small | Very Small |

## Architecture

### PrefixTextEncoder

The `PrefixTextEncoder` class wraps the original text encoder and adds trainable prefix tokens:

1. **Prefix Tokens**: Learnable embeddings that are prepended to the input
2. **Combined Embeddings**: Concatenation of prefix and text embeddings
3. **Attention Mask**: Extended to include prefix tokens
4. **Transformer Processing**: Standard transformer processing with extended sequence
5. **Output**: Returns only the text portion (excluding prefix tokens)

### Training Process

1. **Text Encoding**: Input text is tokenized and embedded
2. **Prefix Addition**: Trainable prefix tokens are prepended
3. **Transformer Processing**: Combined sequence goes through the text encoder
4. **Loss Computation**: Standard diffusion loss is computed
5. **Gradient Update**: Only prefix token parameters are updated

## Tips for Training

1. **Learning Rate**: Start with a higher learning rate (1e-4) since you're only training a small number of parameters
2. **Number of Tokens**: 10-20 tokens usually work well for most tasks
3. **Training Steps**: 2000-4000 steps are typically sufficient
4. **Data Quality**: Ensure your training data is high quality and consistent

## Troubleshooting

### Common Issues

1. **No prefix token parameters found**: Make sure you're loading the correct checkpoint that contains trained prefix tokens
2. **Memory issues**: Reduce batch size or number of prefix tokens
3. **Poor results**: Try increasing the number of prefix tokens or training for more steps

### Debugging

- Check the training logs for loss values and parameter counts
- Verify that only prefix token parameters are being updated
- Monitor the prefix token embeddings during training

## Examples

### Training on Custom Data

```bash
# Train on your own dataset
accelerate launch train_prompt.py \
    --config options/ft_prompt.yml \
    --data_path path/to/your/data.yml \
    --global_batch_size 32
```

### Inference with Different Prompts

```bash
# Generate multiple images with different prompts
python inference_prompt.py \
    --checkpoint_path experiments/ft_prompt/checkpoint-4000/pytorch_model.bin \
    --config_path options/ft_prompt.yml \
    --prompt "A futuristic cityscape at night" \
    --output_path futuristic_city.png

python inference_prompt.py \
    --checkpoint_path experiments/ft_prompt/checkpoint-4000/pytorch_model.bin \
    --config_path options/ft_prompt.yml \
    --prompt "A serene forest with sunlight filtering through trees" \
    --output_path forest.png
```

## References

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [OmniGen2: A Unified Framework for Multi-modal Generation](https://arxiv.org/abs/2401.00000)
