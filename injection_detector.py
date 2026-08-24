import numpy as np
from typing import List, Dict, Any, Tuple


class IntrospectiveInjectionDetector:
    """
    Evaluates model behavior for indirect prompt injections, stealth prompts,
    and topic transition attacks by fusing attention weight patterns and 
    hidden state representations across model layers.
    
    Calibrated for GPT-2 small: focuses on stealth prompt detection via 
    attention entropy and token-level spike anomalies.
    """
    def __init__(
        self, 
        entropy_target: float = 0.42,     # Benign average entropy
        spike_baseline: float = 4.0,       # Benign average spike ratio  
        drift_threshold: float = 0.55,     # Mid-layer drift threshold
        risk_threshold: float = 0.25,      # Detection threshold
        stealth_weight: float = 0.7        # Weight for stealth vs transition
    ):
        self.entropy_target = entropy_target
        self.spike_baseline = spike_baseline
        self.drift_threshold = drift_threshold
        self.risk_threshold = risk_threshold
        self.stealth_weight = stealth_weight

    def calculate_attention_entropy(self, attn_layer: np.ndarray) -> float:
        """Calculates normalized entropy across attention maps."""
        eps = 1e-12
        probs = np.clip(attn_layer, eps, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=-1)
        mean_entropy = np.mean(entropy)
        max_possible_entropy = np.log2(attn_layer.shape[-1] + eps)
        return float(mean_entropy / (max_possible_entropy + eps))

    def calculate_hidden_state_drift(self, hidden_states: List[np.ndarray]) -> List[float]:
        """Measures layer-wise semantic drift (cosine distance) between successive hidden states."""
        drifts = []
        for idx in range(1, len(hidden_states) - 1):
            h_curr = hidden_states[idx][0]
            h_next = hidden_states[idx + 1][0]
            norm_curr = h_curr / (np.linalg.norm(h_curr, axis=-1, keepdims=True) + 1e-10)
            norm_next = h_next / (np.linalg.norm(h_next, axis=-1, keepdims=True) + 1e-10)
            cos_sim = np.sum(norm_curr * norm_next, axis=-1)
            cos_dist = np.mean(1.0 - cos_sim)
            drifts.append(float(cos_dist))
        return drifts

    def calculate_attention_spike_ratio(self, attn_layer: np.ndarray, input_tokens: List[str]) -> Tuple[str, float]:
        """Detects token-level attention spikes, excluding first structural token."""
        head_avg_attn = np.mean(attn_layer, axis=0)
        token_attn_received = np.sum(head_avg_attn, axis=0)
        
        if len(token_attn_received) > 1:
            relevant_tokens = token_attn_received[1:]
            if len(relevant_tokens) > 0:
                max_idx = int(np.argmax(relevant_tokens)) + 1
                mean_attn = np.mean(relevant_tokens)
                spike_ratio = float(relevant_tokens[max_idx - 1] / (mean_attn + 1e-10))
            else:
                max_idx = 0
                spike_ratio = 1.0
        else:
            max_idx = 0
            spike_ratio = 1.0
            
        max_attn_token = input_tokens[max_idx] if max_idx < len(input_tokens) else "N/A"
        return max_attn_token, spike_ratio

    def evaluate_injection_risk(
        self,
        attentions: List[np.ndarray],
        hidden_states: List[np.ndarray],
        input_tokens: List[str]
    ) -> Dict[str, Any]:
        """Combines attention matrices and layer hidden representations for risk assessment."""
        if not attentions or not hidden_states:
            return {"risk_score": 0.0, "status": "SAFE", "details": {}}

        # 1. Multi-Layer Attention Analysis
        num_layers = len(attentions)
        entropies = []
        spike_ratios = []
        spike_tokens = []
        
        for layer_idx in range(num_layers):
            layer_attn = attentions[layer_idx][0]
            entropy = self.calculate_attention_entropy(layer_attn)
            token, spike = self.calculate_attention_spike_ratio(layer_attn, input_tokens)
            entropies.append(entropy)
            spike_ratios.append(spike)
            spike_tokens.append(token)

        avg_entropy = np.mean(entropies)
        max_spike = np.max(spike_ratios)
        max_spike_layer = int(np.argmax(spike_ratios))
        max_spike_token = spike_tokens[max_spike_layer]

        # 2. Hidden State Layer-wise Drift Analysis (mid-layers)
        layer_drifts = self.calculate_hidden_state_drift(hidden_states)
        if layer_drifts and len(layer_drifts) > 4:
            mid_start = min(3, len(layer_drifts) - 2)
            mid_end = min(len(layer_drifts), mid_start + 4)
            mid_drifts = layer_drifts[mid_start:mid_end]
            max_drift_val = float(np.max(mid_drifts)) if mid_drifts else 0.0
            max_drift_layer = int(mid_start + np.argmax(mid_drifts)) + 1
        elif layer_drifts:
            max_drift_layer_rel = int(np.argmax(layer_drifts))
            max_drift_val = layer_drifts[max_drift_layer_rel]
            max_drift_layer = max_drift_layer_rel + 1
        else:
            max_drift_val = 0.0
            max_drift_layer = 0

        # 3. Risk Calculation
        # Stealth Prompt Risk: low entropy + high spikes
        entropy_anomaly = max(0.0, (self.entropy_target - avg_entropy) / self.entropy_target)
        spike_anomaly = max(0.0, (max_spike - self.spike_baseline) / max(self.spike_baseline, 1.0))
        stealth_risk = min(1.0, entropy_anomaly * 0.5 + spike_anomaly * 0.5)
        
        # Topic Transition Risk: mid-layer drift
        transition_risk = min(1.0, max_drift_val / self.drift_threshold) if layer_drifts else 0.0

        # Combined Risk - stealth weighted for indirect injection
        combined_risk_score = float(np.clip(
            self.stealth_weight * stealth_risk + (1 - self.stealth_weight) * transition_risk, 
            0.0, 1.0
        ))

        status = "SUSPICIOUS_PROMPT_INJECTION" if combined_risk_score > self.risk_threshold else "SAFE"

        return {
            "risk_score": round(combined_risk_score, 4),
            "status": status,
            "details": {
                "stealth_prompt_risk": round(float(stealth_risk), 4),
                "topic_transition_risk": round(float(transition_risk), 4),
                "attention_entropy": round(float(avg_entropy), 4),
                "layer_entropies": [round(e, 4) for e in entropies],
                "attention_spike_token": max_spike_token,
                "attention_spike_ratio": round(float(max_spike), 4),
                "max_spike_layer": max_spike_layer,
                "max_layer_drift": round(float(max_drift_val), 4),
                "max_drift_layer": max_drift_layer,
                "layer_drifts": [round(d, 4) for d in layer_drifts],
                "transformer_layers_analyzed": len(layer_drifts)
            }
        }