from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import Client
from typing import Dict


class BaseBench(ABC):
    @abstractmethod
    def evaluate_local_llm(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate a local LLM model.

        Args:
            model (AutoModelForCausalLM): The local LLM model to evaluate.
            tokenizer (AutoTokenizer): The tokenizer for the LLM model.
            subdataset_name (str): The name of the sub-dataset to use for evaluation.
            **kwargs: Additional keyword arguments to pass to the evaluation.

        Returns:
            The evaluation results.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_api_llm(
        self,
        client: Client,
        model: str,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate an API LLM model.

        Args:
            client (Client): The OpenAI client to use for API calls.
            model (str): The name of the API LLM model to evaluate.
            subdataset_name (str): The name of the sub-dataset to use for evaluation.
            **kwargs: Additional keyword arguments to pass to the API call.

        Returns:
            The evaluation results.
        """
        raise NotImplementedError
