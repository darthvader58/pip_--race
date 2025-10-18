import torch, json
from pathlib import Path

ckpt = torch.load("artifacts/rl/qrdqn.pt", map_location="cpu")

class QRDQN(torch.nn.Module):
    def __init__(self, in_dim, n_actions, n_quantiles, hidden):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden, n_actions*n_quantiles)
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
    def forward(self, x):
        z = self.backbone(x)
        q = self.head(z)
        return q.view(-1, self.n_actions, self.n_quantiles)  # (B,2,Q)

net = QRDQN(ckpt["in_dim"], ckpt["n_actions"], ckpt["n_quantiles"], ckpt["hidden"])
net.load_state_dict(ckpt["state_dict"]); net.eval()

dummy = torch.randn(1, ckpt["in_dim"])
torch.onnx.export(
    net, dummy, "artifacts/rl/qrdqn.onnx",
    input_names=["input"], output_names=["qz"],
    dynamic_axes={"input": {0: "batch"}, "qz": {0: "batch"}},
    opset_version=17
)
print("Saved ONNX: artifacts/rl/qrdqn.onnx")
