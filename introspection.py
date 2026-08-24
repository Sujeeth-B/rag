import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any


class ModelIntrospector:
    @staticmethod
    def print_parameter_summary(model):
        print("=== Model Parameters Summary ===")
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            print(f"{name:<60} | Shape: {str(list(param.shape)):<20} | Params: {num_params:,}")
        print("=" * 90)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("=" * 90)

    @staticmethod
    def get_attention_matrix(attentions: List[np.ndarray], layer_idx: int = -1, head_idx: Optional[int] = None) -> np.ndarray:
        # Layer shape: (batch_size, num_heads, seq_len, seq_len)
        layer_attn = attentions[layer_idx][0]  # shape: (num_heads, seq_len, seq_len)
        if head_idx is not None:
            return layer_attn[head_idx]
        return np.mean(layer_attn, axis=0)  # average across heads

    @staticmethod
    def plot_attention_heatmap(
        attentions: List[np.ndarray],
        tokens: List[str],
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        attn_matrix = ModelIntrospector.get_attention_matrix(attentions, layer_idx, head_idx)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            annot=False
        )
        title = f"Attention Weights (Layer {layer_idx}" + (f", Head {head_idx})" if head_idx is not None else ", Avg Heads)")
        plt.title(title)
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved attention heatmap to {save_path}")
        plt.close()

    @staticmethod
    def inspect_hidden_representations(hidden_states: List[np.ndarray]) -> Dict[str, Any]:
        # hidden_states: list of length (num_layers + 1), each element shape (batch_size, seq_len, hidden_dim)
        stats = []
        for l_idx, layer_hs in enumerate(hidden_states):
            layer_data = layer_hs[0]  # (seq_len, hidden_dim)
            stats.append({
                "layer": l_idx,
                "shape": layer_data.shape,
                "mean": float(np.mean(layer_data)),
                "std": float(np.std(layer_data)),
                "min": float(np.min(layer_data)),
                "max": float(np.max(layer_data)),
                "norm_per_token": np.linalg.norm(layer_data, axis=-1).tolist()
            })
        return {"num_layers": len(hidden_states), "layer_stats": stats}
