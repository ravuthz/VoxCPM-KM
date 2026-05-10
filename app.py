import os
import numpy as np
import torch
import gradio as gr
from typing import Optional, Tuple
import voxcpm
from voxcpm import VoxCPM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

VOXCPM_MODEL: Optional[voxcpm.VoxCPM] = None
DEFAULT_MODEL = "openbmb/VoxCPM2"
CACHE_MODEL_DIR = "./data/models/voxcpm/VoxCPM2"


# ---------- Model helpers ----------
def resolve_model_dir() -> str:
    repo_id = DEFAULT_MODEL
    target_dir = CACHE_MODEL_DIR

    if len(repo_id) > 0:
        if not os.path.isdir(target_dir):
            try:
                from huggingface_hub import snapshot_download  # type: ignore

                os.makedirs(target_dir, exist_ok=True)
                print(
                    f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ..."
                )
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                print(f"Warning: HF download failed: {e}. Falling back to 'data'.")
                return "models"
        return target_dir
    return "models"


def get_or_load_voxcpm() -> voxcpm.VoxCPM:
    global VOXCPM_MODEL

    if VOXCPM_MODEL is not None:
        return VOXCPM_MODEL

    print("Model not loaded, initializing...")
    model_dir = resolve_model_dir()
    print(f"Using model dir: {model_dir}")

    VOXCPM_MODEL = voxcpm.VoxCPM(
        voxcpm_model_path=model_dir,
        zipenhancer_model_path="iic/speech_zipenhancer_ans_multiloss_16k_base",
        enable_denoiser=True,
        optimize=False,
    )

    print("Model loaded successfully.")
    return VOXCPM_MODEL


# ---------- Functional endpoints ----------
def prompt_wav_recognition(prompt_wav: Optional[str]) -> str:
    from extensions.builtin.extension_whisper.main import transcribe

    if prompt_wav is None:
        return ""
    result = transcribe(prompt_wav)
    return result


def generate_tts_audio(
    text_input: str,
    prompt_wav_path_input: Optional[str] = None,
    prompt_text_input: Optional[str] = None,
    cfg_value_input: float = 2.0,
    inference_timesteps_input: int = 10,
    do_normalize: bool = True,
    denoise: bool = True,
) -> Tuple[int, np.ndarray]:
    """
    Generate speech from text using VoxCPM; optional reference audio for voice style guidance.
    Returns (sample_rate, waveform_numpy)
    """
    # current_model = get_or_load_voxcpm()
    current_model = VoxCPM.from_pretrained(
        DEFAULT_MODEL,
        load_denoiser=False,
    )

    text = (text_input or "").strip()
    if len(text) == 0:
        raise ValueError("Please input text to synthesize.")

    prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
    prompt_text = prompt_text_input if prompt_text_input else None

    print(f"Generating audio for text: '{text[:60]}...'")

    # wav = current_model.generate(
    #     text=text,
    #     cfg_value=float(cfg_value_input),
    #     inference_timesteps=int(inference_timesteps_input),
    # )

    wav = current_model.generate(
        text=text,
        prompt_text=prompt_text,
        prompt_wav_path=prompt_wav_path,
        cfg_value=float(cfg_value_input),
        inference_timesteps=int(inference_timesteps_input),
        normalize=do_normalize,
        denoise=denoise,
    )

    return (current_model.tts_model.sample_rate, wav)


# ---------- UI Builders ----------


def create_demo_interface():
    """Build the Gradio UI for VoxCPM demo."""
    # Quick Start
    with gr.Accordion("📋 Quick Start Guide", open=False, elem_id="acc_quick"):
        gr.Markdown("""
        ### How to Use
        1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.  
        2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).  
        3. **Enter target text** - Type the text you want the model to speak.  
        4. **Generate Speech** - Click the "Generate" button to create your audio.  
        """)

    # Pro Tips
    with gr.Accordion("💡 Pro Tips", open=False, elem_id="acc_tips"):
        gr.Markdown("""
        ### Prompt Speech Enhancement
        - **Enable** to remove background noise for a clean, studio-like voice, with an external ZipEnhancer component.  
        - **Disable** to preserve the original audio's background atmosphere.  

        ### Text Normalization
        - **Enable** to process general text with an external WeTextProcessing component.  
        - **Disable** to use VoxCPM's native text understanding ability. For example, it supports phonemes input ({HH AH0 L OW1}), try it!  

        ### CFG Value
        - **Lower CFG** if the voice prompt sounds strained or expressive.  
        - **Higher CFG** for better adherence to the prompt speech style or input text.  

        ### Inference Timesteps
        - **Lower** for faster synthesis speed.  
        - **Higher** for better synthesis quality.  
        """)

    # Main controls
    with gr.Row():
        with gr.Column():
            prompt_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Prompt Speech (Optional, or let VoxCPM improvise)",
            )
            DoDenoisePromptAudio = gr.Checkbox(
                value=False,
                label="Prompt Speech Enhancement",
                elem_id="chk_denoise",
                info="We use ZipEnhancer model to denoise the prompt audio.",
            )
            with gr.Row():
                prompt_text = gr.Textbox(
                    label="Prompt Text",
                    placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself...",
                    lines=3,
                )
            run_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            cfg_value = gr.Slider(
                minimum=1.0,
                maximum=3.0,
                value=2.0,
                step=0.1,
                label="CFG Value (Guidance Scale)",
                info="Higher values increase adherence to prompt, lower values allow more creativity",
            )
            inference_timesteps = gr.Slider(
                minimum=4,
                maximum=30,
                value=10,
                step=1,
                label="Inference Timesteps",
                info="Number of inference timesteps for generation (higher values may improve quality but slower)",
            )
            with gr.Row():
                text = gr.Textbox(
                    label="Target Text",
                    value="នៅកម្ពុជា បុណ្យចូលឆ្នាំខ្មែរគឺជាពិធីបុណ្យប្រពៃណីដ៏អស្ចារ្យបំផុត",
                )
            with gr.Row():
                DoNormalizeText = gr.Checkbox(
                    value=False,
                    label="Text Normalization",
                    elem_id="chk_normalize",
                    info="We use wetext library to normalize the input text.",
                )
            audio_output = gr.Audio(label="Output Audio")

    # Wiring
    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            prompt_wav,
            prompt_text,
            cfg_value,
            inference_timesteps,
            DoNormalizeText,
            DoDenoisePromptAudio,
        ],
        outputs=[audio_output],
        show_progress=True,
        api_name="generate",
    )
    prompt_wav.change(
        fn=prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text]
    )


def main():
    with gr.Blocks() as interface:
        create_demo_interface()

    interface.launch(
        server_name=os.environ.get("PORT", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )


if __name__ == "__main__":
    main()
