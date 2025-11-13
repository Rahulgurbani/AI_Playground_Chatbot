
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


_model_cache = {}

def load_model(model_name: str):
    """
    Load and cache a model only once.
    Uses smaller models on CPU to avoid timeouts.
    """
    
    use_gpu = torch.cuda.is_available()

    
    if not use_gpu:
        
        if "llama" in model_name.lower():
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        else:
            model_id = "distilgpt2"  # very light (~300MB)
        print(f"⚙️ Using lightweight model on CPU: {model_id}")
    else:
        
        if model_name.lower() in ["gpt-j", "gptj", "gpt-j-6b"]:
            model_id = "EleutherAI/gpt-j-6B"
        elif "llama" in model_name.lower():
            model_id = "meta-llama/Llama-2-7b-chat-hf"
        else:
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    
    if model_id in _model_cache:
        return _model_cache[model_id]

    print(f"🔄 Loading model: {model_id} (this may take a minute)...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        device_map="auto" if use_gpu else None
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if use_gpu else -1
    )

    _model_cache[model_id] = generator
    print(f"✅ Model loaded successfully: {model_id}")
    return generator


def generate_response(query: str, context: str, llm_name: str = "gpt-j") -> str:
    """
    Generate a text response given a query and retrieved context.
    Automatically trims long outputs and keeps only the answer.
    """
    generator = load_model(llm_name)

    
    prompt = (
        f"Answer the following question using the provided context. "
        f"If the answer is not in the context, give your best explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    outputs = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05
    )

    text = outputs[0]["generated_text"]

    
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    
    if prompt in text:
        text = text.replace(prompt, "").strip()

    
    if len(text) > 1000:
        text = text[:1000] + "..."

    return text
