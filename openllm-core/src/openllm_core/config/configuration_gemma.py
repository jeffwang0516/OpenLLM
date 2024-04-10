from __future__ import annotations

import openllm_core


class GemmaConfig(openllm_core.LLMConfig):
  """Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants.

  Refer to [Gemma's model card](https://ai.google.dev/gemma/docs) for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'url': 'https://ai.google.dev/gemma/docs',
    'architecture': 'GemmaForCausalLM',
    'default_id': 'google/gemma-7b',
    'serialisation': 'safetensors',
    'model_ids': [
        'google/gemma-7b', 'google/gemma-7b-it', 'google/gemma-2b', 'google/gemma-2b-it'
    ]
  }

  class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 12

  class SamplingParams:
    best_of: int = 1
    presence_penalty: float = 0.5
