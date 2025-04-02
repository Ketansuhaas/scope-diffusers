import torch


def slerp(embedding1, embedding2, val):
    """
    usage: slerp(torch.rand(2, 77, 768), torch.rand(2, 77, 768), torch.rand(77))
    """
    original_norm = torch.norm(embedding1)
    low_norm = embedding1 / torch.norm(embedding1)
    high_norm = embedding2 / torch.norm(embedding2)

    print(embedding1.shape, embedding2.shape, val.shape)
    
    dot = (low_norm * high_norm).sum(1)

    omega = torch.acos(dot)
    so = torch.sin(omega)

    faktor1 = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1).unsqueeze(0)

    mask = torch.isnan(faktor1)
    mean = torch.mean(faktor1[~mask])
    faktor1[mask] = mean
    faktor2 = (torch.sin(val * omega) / so).unsqueeze(1).unsqueeze(0)
    mask = torch.isnan(faktor2)
    mean = torch.mean(faktor2[~mask])
    faktor2[mask] = mean

    res = faktor1 * embedding1 + faktor2 * embedding2

    res[0] = res[0] / torch.norm(res[0]) * original_norm
    return res

print(slerp(torch.rand(768), torch.rand(768), torch.rand(1)).shape)