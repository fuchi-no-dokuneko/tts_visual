import os
import sys
import soundfile as sf
import json

# Ensure we are in the batch_eval directory context for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.audio_handler import ReferenceHandler
from utils.report_gen import ReportGenerator
from engines.gsv_engine import GSVEngine

def main():
    # --- Configuration ---
    # Change this to your actual reference directory
    REF_DIR = os.path.expanduser("~/Ver1/ref_wavs")
    OUTPUT_DIR = os.path.abspath("batch_eval/outputs")
    VERSIONS = ["v2", "v2ProPlus", "v3", "v4"]
    
    # Default target text if no specific one is provided
    DEFAULT_TARGET = "こんにちは。これはGPT-SoVITSの各バージョン比較テストです。声の質や自然さを確認してください。"
    
    # Create output structure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Initialization ---
    ref_handler = ReferenceHandler(REF_DIR)
    reporter = ReportGenerator(OUTPUT_DIR)
    
    # We use lazy initialization for the engine
    engine = None
    
    # 1. Scan for reference data
    print(f"[Batch] Scanning references in: {REF_DIR}")
    references = ref_handler.scan_references()
    if not references:
        print("[Error] No valid reference pairs (wav+txt) found in the directory.")
        return
    
    print(f"[Batch] Found {len(references)} characters to process.")
    
    # Results data structure
    # We will enrich the 'references' list with output paths
    results = references 
    
    # 2. Process each model version
    for version in VERSIONS:
        try:
            # Initialize engine on first use or switch
            if engine is None:
                engine = GSVEngine(device="cuda", is_half=True)
            
            engine.load_version(version)
            
            # Directory for this version
            version_subdir = os.path.join(OUTPUT_DIR, version)
            os.makedirs(version_subdir, exist_ok=True)
            
            for ref in results:
                # You can extend this logic to use different target texts per user
                # e.g., if target_texts.get(ref['name']): target = ...
                target_text = DEFAULT_TARGET 
                
                print(f"  > Processing: {ref['name']} ({version})")
                
                out_filename = f"{ref['name']}_{version}.wav"
                out_path = os.path.join(version_subdir, out_filename)
                
                # Skip if already exists (resume support)
                if os.path.exists(out_path):
                    ref[version] = out_path
                    continue
                
                try:
                    sr, audio = engine.infer(
                        ref['wav_path'], 
                        ref['text'], 
                        target_text, 
                        lang="ja"
                    )
                    
                    if audio is not None:
                        # Normalize to int16 for wav file saving
                        if audio.dtype != 'int16':
                            audio = (audio * 32767).astype('int16')
                        sf.write(out_path, audio, sr)
                        ref[version] = out_path
                except Exception as e:
                    print(f"    [!] Error generating {ref['name']}: {e}")
                    
        except Exception as e:
            print(f"[!] Critical error loading version {version}: {e}")
            continue

    # 3. Finalize
    # Save a JSON backup of the results metadata
    with open(os.path.join(OUTPUT_DIR, "results_meta.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # Generate HTML Dashboard
    report_path = reporter.generate_html(results, VERSIONS)
    print(f"\n[Success] All tasks finished.")
    print(f"[Success] Open the report in your browser: file://{report_path}")

if __name__ == "__main__":
    main()
