from transformers import AutoModelForCausalLM

def load_model(model_name="google/gemma-2B"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model

def generate_predictions(model, inputs):
    outputs = model.generate(inputs['input_ids'], max_length=50)
    return outputs
