from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
	model_name: str = Field(default="google/siglip-so400m-patch14-384")
	device: str = Field(default="cuda")
	batch_size: int = Field(default=16)
	batch_max_delay_ms: int = Field(default=8)
	embed_cache_max_items: int = Field(default=5000)
	image_size: int = Field(default=384)
	log_level: str = Field(default="info")
	# Hugging Face auth token (optional) for gated models
	hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
	# Optional Gemma VLM for explanation/summary
	gemma_model_name: str = Field(default="google/gemma-3-2b-it")
	use_gemma_explanations: bool = Field(default=True)
	# Choose primary scorer: 'dual' (SigLIP/CLIP) or 'gemma' (VLM prompt)
	primary_scorer: str = Field(default="dual")

	class Config:
		env_prefix = ""
		case_sensitive = False


settings = Settings()
