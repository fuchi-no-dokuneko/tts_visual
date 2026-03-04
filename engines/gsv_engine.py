import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
except ImportError:
    # Fallback for some environments
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../GPT_SoVITS")))
    from TTS_infer_pack.TTS import TTS, TTS_Config

class GSVEngine:
    def __init__(self, device="cuda", is_half=True):
        self.device = device
        self.is_half = is_half
        self.current_version = None
        self.tts_pipeline = None

    def load_version(self, version):
        if self.current_version == version:
            return
            
        print(f"\n[GSV Engine] Switching to version: {version}...")
        
        weights = {
            "v2": {
                "gpt": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                "sovits": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
            },
            "v2ProPlus": {
                "gpt": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
                "sovits": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
            },
            "v3": {
                "gpt": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
                "sovits": "GPT_SoVITS/pretrained_models/s2Gv3.pth"
            },
            "v4": {
                "gpt": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
                "sovits": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
            }
        }
        
        cfg = weights.get(version)
        if not cfg:
            raise ValueError(f"Unsupported version: {version}")

        # Build config object
        tts_config = TTS_Config({
            "device": self.device,
            "is_half": self.is_half,
            "version": version,
            "t2s_weights_path": cfg["gpt"],
            "vits_weights_path": cfg["sovits"],
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        })
        
        if self.tts_pipeline:
            del self.tts_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        self.tts_pipeline = TTS(tts_config)
        self.current_version = version

    def infer(self, ref_wav, ref_text, target_text, lang="ja"):
        inputs = {
            "text": target_text,
            "text_lang": lang,
            "ref_audio_path": ref_wav,
            "prompt_text": ref_text,
            "prompt_lang": lang,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": 1.0,
            "parallel_infer": True,
            "sample_steps": 32
        }
        
        gen = self.tts_pipeline.run(inputs)
        try:
            sr, audio_data = next(gen)
            return sr, audio_data
        except StopIteration:
            return None, None
