import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import os
import json

def get_model_class(model_id):
    """Determine the appropriate model class based on model ID"""
    return AutoModelForCausalLM

def get_torch_dtype():
    """Get appropriate torch dtype based on GPU capabilities"""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    else:
        return torch.float16

def load_model_and_tokenizer(model_id, use_4bit=True):
    """Load model and tokenizer with quantization for low VRAM usage"""
    # Determine model class
    model_class = get_model_class(model_id)
    
    # Get appropriate dtype
    torch_dtype = get_torch_dtype()
    
    # Model kwargs
    model_kwargs = dict(
        attn_implementation="eager",  # Safer for older GPUs
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    # Add quantization config for 4-bit if requested
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch_dtype,
        )
    
    # Load model and tokenizer
    model = model_class.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set padding side for Gemma models
    tokenizer.padding_side = 'right'
    
    return model, tokenizer

def create_lora_config(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"):
    """Create LoRA configuration"""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

def create_training_args(output_dir="tuned", num_train_epochs=8, per_device_train_batch_size=1,
                        gradient_accumulation_steps=8, learning_rate=1e-4, 
                        gradient_checkpointing=True, fp16=None, bf16=None):
    """Create training arguments"""
    torch_dtype = get_torch_dtype()
    
    # Set precision flags if not provided
    if fp16 is None:
        fp16 = (torch_dtype == torch.float16)
    if bf16 is None:
        bf16 = (torch_dtype == torch.bfloat16)
    
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim="paged_adamw_8bit",
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="tensorboard",
        dataset_kwargs={
            "add_special_tokens": True,
            "append_concat_token": True,
        },
    )

def load_and_prepare_dataset(dataset_path, sample_size=5000, test_size=0.2):
    """Load and prepare dataset for training"""
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Reduce dataset size if needed
    if len(dataset) > sample_size:
        dataset = dataset.select(range(sample_size))
    
    # Split into train/test
    dataset = dataset.train_test_split(test_size=test_size)
    
    return dataset

def train_model(model_id, dataset_path, output_dir="tuned",
                lora_r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                num_train_epochs=8, per_device_train_batch_size=1,
                gradient_accumulation_steps=8, learning_rate=1e-4,
                gradient_checkpointing=True, fp16=None, bf16=None):
    """Main training function"""
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_id)
        
        # Create LoRA config
        peft_config = create_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        
        # Create training args
        training_args = create_training_args(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            bf16=bf16
        )
        
        # Load and prepare dataset
        dataset = load_and_prepare_dataset(dataset_path)
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            peft_config=peft_config
        )
        
        # Start training
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Clean up memory
        del model
        del trainer
        torch.cuda.empty_cache()
        
        return True, f"Training completed successfully. Model saved to {output_dir}"
        
    except Exception as e:
        # Clean up memory in case of error
        torch.cuda.empty_cache()
        return False, f"Training failed with error: {str(e)}"

def merge_lora_model(base_model_id, lora_model_path, output_path="merged_model"):
    """Merge LoRA weights with base model"""
    try:
        # Get appropriate dtype
        torch_dtype = get_torch_dtype()
        
        # Determine model class
        model_class = get_model_class(base_model_id)
        
        # Load base model
        base_model = model_class.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        
        # Load LoRA model
        lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
        
        # Merge models
        merged_model = lora_model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
        tokenizer.save_pretrained(output_path)
        
        # Clean up memory
        del base_model
        del lora_model
        del merged_model
        torch.cuda.empty_cache()
        
        return True, f"Merged model saved to {output_path}"
        
    except Exception as e:
        # Clean up memory in case of error
        torch.cuda.empty_cache()
        return False, f"Model merging failed with error: {str(e)}"