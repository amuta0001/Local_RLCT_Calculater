import numpy as np
import torch


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_tiny_model(model, x_train, y_train, steps=600, lr=5e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()

    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(criterion(model(x_train), y_train).item())
    return final_loss


def make_tiny_regression_data(num_samples=48, dtype=torch.double):
    torch.manual_seed(0)
    np.random.seed(0)

    x_train = torch.randn(num_samples, 2, dtype=dtype)
    y_train = (
        0.7 * x_train[:, :1]
        - 1.2 * x_train[:, 1:2]
        + 0.3 * torch.sin(x_train[:, :1] * x_train[:, 1:2])
    )
    return x_train, y_train


def build_trained_tiny_model(steps=600, lr=5e-2, dtype=torch.double):
    x_train, y_train = make_tiny_regression_data(dtype=dtype)
    model = TinyRegressor().to(dtype=dtype)
    train_loss = train_tiny_model(model, x_train, y_train, steps=steps, lr=lr)
    return model, x_train, y_train, train_loss


def run_tiny_torch_local_test(estimate_rlct_fast_torch_local):
    model, x_train, y_train, train_loss = build_trained_tiny_model()

    criterion = torch.nn.MSELoss()
    result = estimate_rlct_fast_torch_local(
        model=model,
        loss_fn=lambda m, xb, yb: criterion(m(xb), yb),
        x=x_train,
        y=y_train,
        betas=[4, 8, 16],
        step_size=5e-6,
        n_steps=240,
        burn_in=60,
        thinning=6,
        clip_radius=0.15,
        grad_clip=10.0,
        seed=0,
    )

    lambda_hat = result["lambda_hat"]
    beta_ef = result["betaEf"]

    print(f"train_loss={train_loss:.6f}")
    print(f"lambda_hat={lambda_hat:.6f}")
    print(f"betaEf={beta_ef}")
    print(f"scale={result['objective_info']['scale']}")
    print(f"num_params={result['x0'].shape[0]}")

    if not np.isfinite(lambda_hat):
        raise RuntimeError("lambda_hat is not finite.")
    if beta_ef.shape != (3,):
        raise RuntimeError(f"Unexpected betaEf shape: {beta_ef.shape}")

    return result


def main():
    raise RuntimeError(
        "Import run_tiny_torch_local_test from this file and call it from sgld_test.ipynb."
    )


if __name__ == "__main__":
    main()
