import numpy as np
import pytest
import torch

from conformal_explainers.explainers import Explainer
from conformal_explainers.models import SmokeClassifier, build_classifier
from conformal_explainers.training import make_fastshap


@pytest.mark.parametrize(
    "method", ["saliency", "input_x_gradient", "gradient_shap", "kernel_shap"]
)
def test_real_attribution_backends(method):
    torch.manual_seed(2)
    np.random.seed(2)
    x = torch.ones(3, 8, 8)
    attr = Explainer(SmokeClassifier(), method, samples=32, grid=2)(x, 1)
    assert attr.shape == (8, 8)
    assert np.isfinite(attr).all()
    assert attr[:4, :4].mean() > attr[4:, 4:].mean()


def test_missing_fastshap_checkpoint_fails_instead_of_fake_explanation():
    with pytest.raises(ValueError, match="trained checkpoint"):
        Explainer(SmokeClassifier(), "fastshap")


def test_fastshap_training_and_efficiency():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    f = make_fastshap(SmokeClassifier(), 8, 2, 2, torch.device("cpu"))
    train, val = torch.rand(8, 3, 8, 8), torch.rand(4, 3, 8, 8)
    before = [p.detach().clone() for p in f.explainer.parameters()]
    f.train(
        train,
        val,
        batch_size=4,
        num_samples=4,
        max_epochs=1,
        validation_samples=4,
        training_seed=42,
        validation_seed=43,
    )
    assert any(not torch.equal(a, b) for a, b in zip(before, f.explainer.parameters()))
    a = f.shap_values(val)
    expected = SmokeClassifier()(val).softmax(1) - SmokeClassifier()(
        torch.zeros_like(val)
    ).softmax(1)
    np.testing.assert_allclose(a.sum((2, 3)), expected.numpy(), atol=1e-5)


def test_resnet_training_backward_without_network_download():
    torch.set_num_threads(2)
    m = build_classifier(10, pretrained=False)
    x = torch.rand(2, 3, 32, 32)
    loss = torch.nn.functional.cross_entropy(m(x), torch.tensor([0, 1]))
    loss.backward()
    assert m.fc.weight.grad is not None and torch.isfinite(loss)
