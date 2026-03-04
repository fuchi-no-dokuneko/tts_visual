import os
import wave
import contextlib

class ReferenceHandler:
    def __init__(self, ref_dir):
        self.ref_dir = ref_dir

    def get_audio_duration(self, file_path):
        try:
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0

    def scan_references(self):
        """Scan directory and return matched wav/txt pairs."""
        refs = []
        if not os.path.exists(self.ref_dir):
            return refs
            
        files = os.listdir(self.ref_dir)
        wav_files = [f for f in files if f.endswith('.wav')]
        
        for wav in sorted(wav_files):
            name = os.path.splitext(wav)[0]
            txt_file = name + ".txt"
            wav_path = os.path.abspath(os.path.join(self.ref_dir, wav))
            txt_path = os.path.abspath(os.path.join(self.ref_dir, txt_file))
            
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                except:
                    text = ""
                
                duration = self.get_audio_duration(wav_path)
                # Quality recommendation: 3-10s
                status = "OK" if 2.8 <= duration <= 12.0 else "Length Issue"
                
                refs.append({
                    "name": name,
                    "wav_path": wav_path,
                    "text": text,
                    "duration": round(duration, 2),
                    "status": status
                })
        return refs
