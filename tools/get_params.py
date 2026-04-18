import os
import torch


def load_state_dict_from_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']

    if isinstance(checkpoint, dict):
        return checkpoint

    raise TypeError(f'Unsupported checkpoint format: {type(checkpoint)}')


def count_params_from_checkpoint(ckpt_path):
    state_dict = load_state_dict_from_checkpoint(ckpt_path)

    total_params = sum(
        tensor.numel() for tensor in state_dict.values()
        if torch.is_tensor(tensor)
    )

    approx_size_mb = total_params * 4 / 1024 / 1024

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Total params: {total_params:,}")
    print(f"Approx weight size (FP32): {approx_size_mb:.2f} MB")


if __name__ == "__main__":
    checkpoints = [
        "work_dirs/unet_tongji/iter_8000.pth",
        "work_dirs/deeplabv3plus_tongji/iter_8000.pth",
        "work_dirs/pspnet_tongji/iter_8000.pth",
        "work_dirs/segformer_tongji/iter_8000.pth",
    ]

    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            count_params_from_checkpoint(ckpt)
        else:
            print(f"\nCheckpoint not found: {ckpt}")