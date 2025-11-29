from .base import BaseModel
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class MiniEmo(BaseModel):
    def __init__(self, model_path, **kwargs):
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",
            )
            .cuda()
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        kwargs_default = {"max_new_tokens": 64, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def generate_inner(self, message, dataset=None):

        message_text = None
        message_image = None

        for m in message:
            if m["type"] == "text":
                message_text = m["value"]
            elif m["type"] == "image":
                pil_rgb = Image.open(m["value"]).convert("RGB")
                message_image = pil_rgb
                
        def prepare_inputs(prompt_text: str):
            final_prompt = build_miniemo_prompt(self.model.config, prompt_text)
            encoded = self.processor(
                text=final_prompt,
                images=message_image,
                return_tensors="pt",
                padding=True,
            )
            device = next(self.model.parameters()).device
            encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
            return encoded, final_prompt

        inputs, final_prompt = prepare_inputs(message_text)
        
        # print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        # print(f"   Pixel values shape: {inputs['pixel_values'].shape}")
        
        generation_kwargs = dict(
            max_new_tokens=160,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        inputs, final_prompt = prepare_inputs(message_text)
        with torch.inference_mode():
            outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask"),
                    **generation_kwargs,
            )
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = postprocess_generation(raw_response, final_prompt)
        if not response:
            response = "(empty response)"

        if self.kwargs.get("verbose", False):
            print(f"   [MiniEmo] response: {response}")
        return response


DEFAULT_EMOTION_CATEGORIES = (
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
)


def _strip_special_tokens(config, text: str) -> str:
    for token in (
        getattr(config, "image_token", None),
        getattr(config, "sentiment_token", None),
        getattr(config, "emotion_token", None),
    ):
        if token:
            text = text.replace(token, "")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def build_miniemo_prompt(config, user_prompt: str) -> str:
    raw_categories = getattr(config, "categories", None)
    if isinstance(raw_categories, dict) and raw_categories:
        category_names = list(raw_categories.keys())
    elif isinstance(raw_categories, (list, tuple)) and raw_categories:
        category_names = list(raw_categories)
    else:
        category_names = list(DEFAULT_EMOTION_CATEGORIES)

    category_str = ", ".join(category_names)
    instruction_template = getattr(config, "instruction_prompt", None)
    system_prompt = getattr(config, "system_prompt", None)

    cleaned_default = None
    if instruction_template:
        try:
            cleaned_default = _strip_special_tokens(config, instruction_template.format(category_str))
        except Exception:
            cleaned_default = _strip_special_tokens(config, instruction_template)

    user_prompt = (user_prompt or "").strip()
    if user_prompt:
        question = user_prompt
        if not question.endswith((".", "?", "!")):
            question = f"{question}."
        final_prompt = (
            f"{question}\n"
            "Answer using the visual content of the image and mention the key cues that support your reply."
        )
    else:
        final_prompt = cleaned_default or (
            "Focus on visual evidence (e.g., facial expressions, objects, scene, visual attributes). "
            f"Choose the emotion category that best represents the image from one of the following categories: {category_str}. "
            "Provide an explanation of your choice based on the visual evidence.\n"
            "Predicted Label: <category>.\nJustification: <short paragraph>."
        )

    if system_prompt:
        system_prefix = _strip_special_tokens(config, system_prompt)
        if system_prefix:
            final_prompt = f"{system_prefix}\n\n{final_prompt}".strip()

    return final_prompt


def postprocess_generation(text: str, prompt: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    if prompt and text.startswith(prompt):
        trimmed = text[len(prompt):].strip()
        if trimmed:
            text = trimmed
    if text.startswith('"') and text.endswith('"') and len(text) > 1:
        text = text[1:-1].strip()
    return text or "(empty response)"
