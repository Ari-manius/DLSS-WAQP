from torch_geometric.nn import GCNConv, SAGEConv
from torch import nn

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, GCNConv):
            if hasattr(m, 'lin') and m.lin is not None:
                if hasattr(m.lin, 'weight'):
                    nn.init.kaiming_uniform_(m.lin.weight)
                if hasattr(m.lin, 'bias') and m.lin.bias is not None:
                    nn.init.zeros_(m.lin.bias)

        elif isinstance(m, SAGEConv):
            if hasattr(m, 'lin_l') and m.lin_l is not None:
                nn.init.kaiming_uniform_(m.lin_l.weight)
                if m.lin_l.bias is not None:
                    nn.init.zeros_(m.lin_l.bias)
            if hasattr(m, 'lin_r') and m.lin_r is not None:
                nn.init.kaiming_uniform_(m.lin_r.weight)
                if m.lin_r.bias is not None:
                    nn.init.zeros_(m.lin_r.bias)

