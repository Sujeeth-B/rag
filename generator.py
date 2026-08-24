import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Tuple, List, Optional


class IntrospectiveGenerator:
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True
        ).to(self.device)
        self.model.eval()

    def generate_with_introspection(
        self, prompt: str, max_new_tokens: int = 30
    ) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True
            )

        # Forward pass introspection on full prompt input
        # outputs.attentions: tuple of (batch_size, num_heads, seq_len, seq_len) per layer
        # outputs.hidden_states: tuple of (batch_size, seq_len, hidden_dim) for embeddings + each layer
        attentions = [att.cpu().numpy() for att in outputs.attentions] if outputs.attentions else []
        hidden_states = [hs.cpu().numpy() for hs in outputs.hidden_states] if outputs.hidden_states else []

        # Autoregressive generation
        with torch.no_grad():
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        generated_ids = gen_output.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        gen_tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "input_tokens": tokens,
            "all_tokens": gen_tokens,
            "attentions": attentions,  # List of numpy arrays: [layer_idx] -> shape (1, num_heads, seq_len, seq_len)
            "hidden_states": hidden_states,  # List of numpy arrays: [layer_idx] -> shape (1, seq_len, hidden_dim)
            "gen_output": gen_output
        }
