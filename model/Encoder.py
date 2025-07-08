import torch
import torch.nn as nn

class LinearProj(nn.Module):
    def __init__(self, cfg):
        super(LinearProj, self).__init__()
        self.mapper = nn.Linear(cfg.model.text_dimension, cfg.model.vision_dimension)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, cfg.model.text_dimension]
        Returns:
            out: Tensor of shape [batch_size, cfg.model.vision_dimension]
        """
        return self.mapper(x)

def train_proj(cfg, model, dataloader, criterion, device='cuda'):
    """
    Train the Projector.

    Args:
        model: the LinearProj model
        dataloader: PyTorch DataLoader yielding (input, target) tensors
        device: 'cuda' or 'cpu'
    """
    num_epochs = cfg.model.num_epochs
    lr = cfg.model.lr_proj

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)  # [B, text_dimension]
            y = y.to(device)  # [B, vision_dimension]

            optimizer.zero_grad()
            output = model(x)  # [B, vision_dimension]
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.6f}")

    return model
