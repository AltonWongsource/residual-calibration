import torch

from train_time_cpu.losses import LossConfig, MDCALoss, MbLSLoss, build_loss


def test_losses_are_finite_and_differentiable():
    logits = torch.tensor([[2.0, 0.0], [0.2, 1.2], [0.5, 0.5]], requires_grad=True)
    targets = torch.tensor([0, 1, 0])
    for name in ("ce", "mdca", "mbls"):
        loss = build_loss(LossConfig(name=name, num_classes=2))(logits, targets)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)


def test_mdca_zero_for_matching_batch_mass():
    loss = MDCALoss(num_classes=2)
    logits = torch.zeros(4, 2)
    targets = torch.tensor([0, 0, 1, 1])
    assert torch.allclose(loss.mdca_term(logits, targets), torch.tensor(0.0))


def test_mbls_penalizes_excess_margin():
    loss = MbLSLoss(margin=1.0, penalty_weight=0.1)
    logits = torch.tensor([[4.0, 0.0]])
    assert float(loss.penalty(logits)) > 0.0
