import os
import sys

class GSVEngine:
    def __init__(self, root, weights, device="cpu", precision="float32", seed=0):
        self.root = os.path.abspath(root)
        self.weights = weights
        self.device = device
        self.is_half = precision == "float16"
        self.seed = seed
        self.current_version = None
        self.tts_pipeline = None

        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        import torch
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        self.torch = torch
        self.TTS = TTS
        self.TTS_Config = TTS_Config
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_version(self, version):
        if self.current_version == version:
            return
            
        print(f"\n[GSV Engine] Switching to version: {version}...")
        
        cfg = self.weights.get(version)
        if not cfg:
            raise ValueError(f"Unsupported version: {version}")

        def weight_path(value):
            return value if os.path.isabs(value) else os.path.join(self.root, value)

        tts_config = self.TTS_Config({
            "device": self.device,
            "is_half": self.is_half,
            "version": version,
            "t2s_weights_path": weight_path(cfg["gpt"]),
            "vits_weights_path": weight_path(cfg["sovits"]),
            "cnhuhbert_base_path": os.path.join(self.root, "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
            "bert_base_path": os.path.join(self.root, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        })
        
        if self.tts_pipeline:
            del self.tts_pipeline
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            
        self.tts_pipeline = self.TTS(tts_config)
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
