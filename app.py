import os
import gradio as gr
from transformers import pipeline

HF_API = os.environ.get("HF_API")

model = "aegrif/gpt2_spell_gen"
tokenizer = "aegrif/gpt2_spell_gen"
pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, use_auth_token=HF_API)


def predict(text, temperature, top_k, top_p, max_length):
    input_text = f"<|name|> {text} <|spell|>"
    pipeline.model.config.pad_token_id = pipeline.model.config.eos_token_id
    pipeline.model.config.temperature = temperature
    pipeline.model.config.top_k = top_k
    pipeline.model.config.top_p = top_p
    predictions = pipeline(input_text, max_length=max_length, num_return_sequences=1)[0]["generated_text"]
    spell_start = len(text) + 19
    output = text + "\n\n" + predictions[spell_start:]
    return output.strip()


title = "# Spell generation with GPT-2"
description = "## Generate your own spells"
examples = [["Speak with Objects"], ["Summon Burley"], ["Moon Step"], ["Burden of the Gods"], ["Shape Rock"], ["Bard's Laughter"], ["Mundane Foresight"], ["Word of Cancellation"]]

with gr.Blocks(css="#spell-row {justify-content: flex-start; }") as interface:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row(variant="compact", elem_id="spell-row"):
        with gr.Column(scale=1):
            name = gr.Textbox(lines=1, label="Spell Name", placeholder="Enter your spell name here...")
    with gr.Row(variant="compact", elem_id="spell-row"):
        with gr.Column(scale=3):
            gr.Examples(examples, inputs=name)
    with gr.Row(variant="compact", elem_id="spell-row"):
        gr.Markdown("## Model Settings")
    with gr.Row(variant="compact", elem_id="spell-row"):
        with gr.Column(scale=1):
            max_length = gr.Slider(minimum=50, maximum=800, step=50, value=400, label="Max Length")
        with gr.Column(scale=1):
            temperature = gr.Slider(minimum=0.1, maximum=float(1.9), step=0.1, value=float(1), label="Temperature")
    with gr.Row(variant="compact", elem_id="spell-row"):
        with gr.Column(scale=1):
            top_k = gr.Slider(minimum=0, maximum=1000, step=10, value=50, label="Top K")
        with gr.Column(scale=1):
            top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1, label="Top P")
    with gr.Row(variant="compact", elem_id="spell-row"):
        output = gr.Textbox(label="Generated Spell", placeholder="Your spell will appear here.")
    generate_btn = gr.Button("Generate Spell")
    with gr.Row(variant="compact", elem_id="spell-row"):
        gr.Markdown(
            "**Max Length**: The maximum length of the generated spell.\n\n**Temperature**: The randomness of the generated spell. Higher values are more random, lower values are more deterministic.\n\n**Top K**: The number of highest probability vocabulary tokens to keep for top-k-filtering.\n\n**Top P**: The cumulative probability for top-p-filtering.")

    generate_btn.click(fn=predict, inputs=[name,temperature,top_k,top_p, max_length], outputs=output)

interface.launch()