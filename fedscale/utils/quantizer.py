import torch

def qsgd_quantize(tensor, bits=4):
    """Simple QSGD-style quantizer as mentioned in the paper."""
    if bits >= 32: return tensor
    s = 2 ** bits
    norm = torch.norm(tensor)
    if norm == 0: return tensor
    
    level_float = s * torch.abs(tensor) / norm
    previous_level = torch.floor(level_float)
    is_next_level = (torch.rand_like(tensor) < (level_float - previous_level)).float()
    
    quantized = norm * torch.sign(tensor) * (previous_level + is_next_level) / s
    return quantized
