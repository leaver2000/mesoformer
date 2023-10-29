import torch
from model.mae import MaskedAutoencoder3d


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mae() -> None:
    B = 10
    C = 5
    T = 8
    # Z = 12
    H = 40
    W = 80
    input_shape = (T, H, W)
    patch_shape = (4, 4, 8)
    model = (
        MaskedAutoencoder3d(
            input_shape, patch_shape, batch_size=B, in_chans=C, num_heads=12, embed_dim=768, decoder_embed_dim=768
        )
        .train(True)
        .to(device)
    )
    batch = torch.randn(B, C, T, H, W).to(device)

    loss, pred, mask = model(batch)

    assert pred.shape == (10, 200, 640)
    assert model.from_patches(pred).shape == (B, C, T, H, W)
