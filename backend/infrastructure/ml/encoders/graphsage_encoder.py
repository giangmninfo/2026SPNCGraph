import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(torch.nn.Module):
    """
    GraphSAGE-based feature encoder.

    Used ONLY to project node features into the same latent
    space as training nodes for kNN inference.

    Output:
        Tensor shape (1, hidden_dim)
    """

    def __init__(
        self,
        in_dim: int = 896,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (1, in_dim)

        Returns:
            Tensor (1, hidden_dim), L2-normalized
        """
        x = x.to(self.device)

        # No neighbors at inference → projection only
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=self.device
        )

        z = self.conv1(x, edge_index)
        z = F.relu(z)
        z = F.normalize(z, dim=1)

        return z.cpu()
    
    def forward(self, x):
        # no edges for inductive single-node encoding
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=x.device
        )
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return F.normalize(x, dim=1)