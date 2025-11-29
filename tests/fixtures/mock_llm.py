"""Mock LLM implementations for testing without requiring actual model loading."""

from typing import Dict, List, Any, Optional
import random
import string


class MockTokenizer:
    """Mock tokenizer that simulates transformers.AutoTokenizer behavior."""
    
    def __init__(self, max_length: int = 4096):
        self.max_length = max_length
        self.eos_token_id = 2
        self.pad_token_id = 0
        
    def __call__(self, text: str, return_tensors: str = None, truncation: bool = False, max_length: int = None, **kwargs):
        """Tokenize text and return mock tensor-like object."""
        max_len = max_length or self.max_length
        if truncation and len(text) > max_len:
            text = text[:max_len]
        
        # Create mock tensors
        input_ids = [1] * min(len(text) // 10, 100) + [2]  # Simulate token ids
        attention_mask = [1] * len(input_ids)
        
        if return_tensors == "pt":
            return {
                "input_ids": MockTensor(input_ids),
                "attention_mask": MockTensor(attention_mask),
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        """Decode token ids back to text."""
        # Return a simulated response based on token count
        response_length = len(token_ids) if isinstance(token_ids, list) else 100
        if hasattr(token_ids, 'tolist'):  # Handle tensor-like objects
            response_length = len(token_ids.tolist())
        
        # Generate a deterministic fake response based on token_ids
        random.seed(sum(token_ids.tolist()) if hasattr(token_ids, 'tolist') else 42)
        words = [
            'The', 'answer', 'to', 'this', 'question', 'is', 'based', 'on',
            'the', 'provided', 'context.', 'According', 'to', 'the', 'document,',
            'this', 'is', 'a', 'test', 'response.', 'The', 'model', 'generates',
            'this', 'output', 'based', 'on', 'the', 'input', 'tokens.'
        ]
        num_words = min(response_length // 5, 50)
        response = ' '.join(random.choices(words, k=num_words))
        return response


class MockTensor:
    """Mock tensor object that behaves like PyTorch tensor."""
    
    def __init__(self, data: List, device: str = "cpu"):
        self.data = data
        self.device = device
        self.dtype = "float32"
    
    def to(self, device: str):
        """Move tensor to device (mock implementation)."""
        return MockTensor(self.data, device)
    
    def tolist(self):
        """Convert to list."""
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """Support indexing to get individual elements or slices."""
        return self.data[index]
    
    def __repr__(self):
        return f"MockTensor({self.data})"


class MockModel:
    """Mock language model that simulates transformers.AutoModelForCausalLM."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.device = "cpu"
        self.config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "vocab_size": 50257,
        }
        self._init_responses()
    
    def _init_responses(self):
        """Initialize deterministic mock responses."""
        self.responses = {
            "short": "This is a short answer.",
            "medium": "This is a more detailed answer that provides more context and information about the topic.",
            "long": "This is a comprehensive answer that thoroughly explains the concept. It includes multiple sentences and provides detailed information based on the context provided in the input.",
        }
    
    def generate(self, input_ids: MockTensor = None, attention_mask: MockTensor = None, 
                 max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9,
                 **kwargs) -> MockTensor:
        """Generate mock response tokens."""
        # Simulate token generation
        input_len = len(input_ids.data) if hasattr(input_ids, 'data') else 100
        num_new_tokens = min(max_new_tokens, random.randint(50, 200))
        
        # Create output with input tokens + new tokens
        output = input_ids.data + [1] * num_new_tokens + [2]
        return MockTensor(output)
    
    def to(self, device: str):
        """Move model to device (mock implementation)."""
        self.device = device
        return self


class MockAPIResponse:
    """Mock OpenAI API response."""
    
    def __init__(self, content: str):
        self.content = content
    
    def __repr__(self):
        return f"MockMessage(content={self.content})"


class MockAPIChoice:
    """Mock API choice/message wrapper."""
    
    def __init__(self, content: str):
        self.message = MockAPIMessage(content)
    
    def __repr__(self):
        return f"MockChoice(message={self.message})"


class MockAPIMessage:
    """Mock API message object."""
    
    def __init__(self, content: str):
        self.content = content
    
    def __repr__(self):
        return f"MockMessage(content={self.content})"


class MockAPIResponse:
    """Mock OpenAI API completion response."""
    
    def __init__(self, content: str):
        self.choices = [MockAPIChoice(content)]
    
    def __repr__(self):
        return f"MockAPIResponse(choices={self.choices})"


class _MessagesProxy:
    """Proxy to support client.messages.create() API style."""
    
    def __init__(self, client):
        self.client = client
    
    def create(self, *args, **kwargs):
        return self.client.messages_create(*args, **kwargs)


class MockOpenAIClient:
    """Mock OpenAI Client for testing without API calls."""
    
    def __init__(self, api_key: str = "mock-key", base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self._response_counter = 0
        self._messages = _MessagesProxy(self)
    
    def messages_create(self, model: str = None, messages: List[Dict] = None, 
                       max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> MockAPIResponse:
        """Mock the messages.create API call."""
        # Extract the user message
        user_message = ""
        if messages:
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
        
        # Generate a deterministic response based on message content
        response_length = len(user_message) // 20
        if response_length < 20:
            response = "Yes, that's correct. This is a test response."
        elif response_length < 50:
            response = "This is a medium-length response that provides some additional context about the topic mentioned in your question."
        else:
            response = "This is a comprehensive and detailed response that thoroughly addresses all aspects of the question. It includes relevant information from the provided context and additional insights. The answer demonstrates understanding of the underlying concepts."
        
        self._response_counter += 1
        return MockAPIResponse(response)
    
    # Support both old and new API styles
    def create(self, *args, **kwargs):
        """Support legacy create method for compatibility."""
        return self.messages_create(*args, **kwargs)
    
    @property
    def messages(self):
        """Support the client.messages.create() style API."""
        return self._messages


class MockResponse:
    """Mock response from mocked messages object."""
    
    def __init__(self, content: str):
        self.choices = [MockAPIChoice(content)]


def create_mock_tokenizer(max_length: int = 4096) -> MockTokenizer:
    """Factory function to create mock tokenizer."""
    return MockTokenizer(max_length=max_length)


def create_mock_model(model_name: str = "mock-model") -> MockModel:
    """Factory function to create mock model."""
    return MockModel(model_name=model_name)


def create_mock_client(api_key: str = "mock-key", base_url: str = None) -> MockOpenAIClient:
    """Factory function to create mock OpenAI client."""
    return MockOpenAIClient(api_key=api_key, base_url=base_url)
