import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple, List, Dict, Any

class MiniEngine:
    """
    Manual inference engine for LLM inference on Apple Silicon.
    Implements custom decoding loop with KV cache management.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "mps"):
        """
        Initialize the MiniEngine with a model and tokenizer.
        
        Args:
            model_name: Hugging Face model identifier (e.g., "gpt2")
            device: Device to run inference on
        """
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad_token if None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)

        # Move model to device
        self.model = self.model.to(device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Store configuration values
        self.vocab_size = self.config.vocab_size
        self.hidden_size = getattr(self.config, 'hidden_size', getattr(self.config, 'n_embd', None))
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"Model {model_name} loaded on {device}")
    
    def generate_token(
        self, 
        input_ids: torch.Tensor, 
        cache: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        Perform a single forward pass and generate the next token.
        
        Args:
            input_ids: Input token IDs tensor of shape [batch_size, seq_len]
            cache: Optional KV cache from previous forward passes
            temperature: Sampling temperature (1.0 = no change, >1.0 = more random)
        
        Returns:
            next_token_id: The generated token ID (as Python int)
            updated_cache: Updated KV cache dictionary or tuple
        """
        # Move input_ids to device
        input_ids = input_ids.to(self.device)
        
        # Handle cache - if cache exists, you might only need last token
        if cache is not None:
            input_ids = input_ids[:, -1:]
        
        # Prepare model inputs
        model_inputs = {"input_ids": input_ids}
        
        # Add past_key_values if cache exists
        if cache is not None:
            model_inputs["past_key_values"] = cache
        
        # Forward pass (use torch.no_grad() for efficiency)
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        # Extract logits from last position
        logits = outputs.logits[:, -1, :]
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample next token (use multinomial for sampling or argmax for greedy)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        # Extract updated cache from outputs
        updated_cache = outputs.past_key_values
        
        # Return token ID (as int) and updated cache
        return next_token_id, updated_cache
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100, 
        temperature: float = 0.7,
        stop_on_eos: bool = True
    ) -> str:
        """
        Generate text from a prompt using manual inference loop.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stop_on_eos: Whether to stop generation when EOS token is encountered
        
        Returns:
            Generated text (including the prompt)
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Initialize cache
        cache = None
        
        # Store all token IDs
        generated_token_ids = input_ids[0].tolist()
        
        # Generation loop
        for step in range(max_tokens):
             # Generate next token
            next_token_id, cache = self.generate_token(input_ids, cache, temperature)
             
             # Add to sequence
            generated_token_ids.append(next_token_id)
             
             # Update input_ids for next iteration
             # (If cache exists, you only need the last token)
            if cache is not None:
                input_ids = torch.tensor([[next_token_id]], device=self.device)
            else:
                 input_ids = torch.tensor([[next_token_id]], device=self.device)
             
             # Check for EOS token
            if stop_on_eos and next_token_id == self.eos_token_id:
                break
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # Return generated text
        return generated_text
    
    def generate_streaming(
        self, 
        prompt: str, 
        max_tokens: int = 100, 
        temperature: float = 0.7,
        stop_on_eos: bool = True
    ):
        """
        Generator that yields tokens as they are generated (for streaming).
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
        
        Yields:
            Generated token text (one token at a time)
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        cache = None
        
        for step in range(max_tokens):
            next_token_id, cache = self.generate_token(input_ids, cache, temperature)
            yield self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            if cache is not None:
                input_ids = torch.tensor([[next_token_id]], device=self.device)
            else:
                input_ids = torch.tensor([[next_token_id]], device=self.device)
            if stop_on_eos and next_token_id == self.eos_token_id:
                break  