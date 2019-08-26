import torch

def texts_to_tensor(ids, max_len):
    result = torch.zeros(len(ids), max_len, dtype=torch.long)
    for i, row in enumerate(ids):
        result[i,:min(len(row), max_len)] = torch.tensor(row[:max_len])
    return result