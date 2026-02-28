
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


# ---------------- DATA GENERATION ----------------#

def generate_sample(size=100):
    grid = np.zeros((size, size), dtype=np.uint8)

    # Random rectangles
    for _ in range(np.random.randint(3, 8)):
        x = np.random.randint(0, size - 10)
        y = np.random.randint(0, size - 10)
        w = np.random.randint(5, 25)
        h = np.random.randint(5, 25)
        grid[y: y +h, x: x +w] = 1

    grid = binary_dilation(grid, iterations=1).astype(np.uint8)

    # Vertical wall
    if np.random.rand() < 0.4:
        x = np.random.randint(10, size - 10)
        grid[:, x: x +3] = 1

    # Horizontal wall
    if np.random.rand() < 0.4:
        y = np.random.randint(10, size - 10)
        grid[y: y +3, :] = 1

    # Distance transforms
    obs_dist = distance_transform_edt(1 - grid)

    yy, xx = np.meshgrid(
        np.arange(size),
        np.arange(size),
        indexing="ij"
    )

    boundary_dist = np.minimum.reduce([
        yy,
        xx,
        size - 1 - yy,
        size - 1 - xx
    ])

    # Normalize input channels
    obs_dist_norm = obs_dist / (obs_dist.max() + 1e-6)
    boundary_dist_norm = boundary_dist / (boundary_dist.max() + 1e-6)

    # Risk field (target)
    cost = (
            3.5 * np.exp(-obs_dist / 1.8) +
            0.3 * np.exp(-boundary_dist / 10)
    )



    cost = cost / cost.max()

    # 3-channel input:
    # [obstacle grid, obstacle distance, boundary distance]
    input_tensor = np.stack(
        [grid.astype(np.float32),
         obs_dist_norm.astype(np.float32),
         boundary_dist_norm.astype(np.float32)],
        axis=0
    )

    return input_tensor, cost.astype(np.float32)


# ---------------- MODEL ----------------#

# ---------------- CNN ----------------#
class CostCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder ----- [User's wish what to use , I have used u-net style architecture]
        # ----- Bottleneck -----
        # ----- Decoder -----


    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        return self.out(d1)



# ---------------- TRAINING ----------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CostCNN().to(device)



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )


    # loss_fn = nn.MSELoss()  [mse]
    # Huber loss
    loss_fn = nn.SmoothL1Loss() # sharp edge but mse works better too


# tried with different epochs , [After certain point:Loss reduction becomes negligible] so went with 2k
    for epoch in range(2000):

        batch_x = []
        batch_y = []

        # One size per batch
        size = np.random.choice([64, 96, 128])



        for _ in range(16):
            inp, cost = generate_sample(size)
            batch_x.append(inp)
            batch_y.append(cost)

        batch_x = np.array(batch_x)      # (B, 3, H, W)
        batch_y = np.array(batch_y)      # (B, H, W)

        x = torch.tensor(batch_x).to(device)
        y = torch.tensor(batch_y).unsqueeze(1).to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.6f}")

    torch.save(model.state_dict(), "cnn_cost.pth")
    print("Model saved as cnn_cost.pth")



if __name__ == "__main__":
    train()
