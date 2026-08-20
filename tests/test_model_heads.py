import torch

from hydroproto import HydroLocalHead, HydroMPC650MHead, HydroProtoHead, SequenceMLP


def test_head_output_shapes():
    x = torch.randn(4, 1280)
    assert SequenceMLP(1280, 512, 256, 5, 0.2)(x).shape == (4, 5)
    assert HydroLocalHead(1280, 512, 256, 128, 5, 0.2)(x)["logits"].shape == (4, 5)
    args = (1280, 512, 256, 128, 5, 5, 0.1, 0.2)
    assert HydroProtoHead(*args)(x)["class_logits"].shape == (4, 5)
    assert HydroMPC650MHead(*args)(x)["class_logits"].shape == (4, 5)
