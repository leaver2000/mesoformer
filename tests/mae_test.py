import torch
from model.mae import MaskedAutoencoder3d, MaskedAutoencoder4d


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mae_3d() -> None:
    B = 10
    C = 5
    T = 8
    # Z = 12
    H = 40
    W = 80
    input_shape = (T, H, W)
    patch_shape = (4, 4, 8)
    model = (
        MaskedAutoencoder3d(B, C, input_shape, patch_shape, num_heads=12, embed_dim=768, decoder_embed_dim=768)
        .train(True)
        .to(device)
    )
    batch = torch.randn(B, C, T, H, W).to(device)
    assert torch.all(model.patch_decode(model.patch_encode(batch)) == batch).item()
    loss, pred, mask = model.__call__(batch)

    assert pred.shape == (10, 200, 640)
    assert model.patch_decode(pred).shape == (B, C, T, H, W)


def test_mae_4d() -> None:
    B = 10
    C = 5
    T = 8
    Z = 12
    Y = 40
    X = 80
    input_shape = (T, Z, Y, X)
    patch_shape = (4, 4, 4, 8)
    model = (
        MaskedAutoencoder4d(B, C, input_shape, patch_shape, num_heads=12, embed_dim=768, decoder_embed_dim=768)
        .train(True)
        .to(device)
    )
    batch = torch.randn(B, C, T, Z, Y, X).to(device)
    assert torch.all(model.patch_decode(model.patch_encode(batch)) == batch).item()
    loss, pred, mask = model.__call__(batch)
