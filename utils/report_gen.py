import json
import os

class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_html(self, results, versions):
        """
        results: list of dicts {name, wav_path, text, v2: path, v3: path, ...}
        versions: list of versions compared
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TTS Model Comparison Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; margin: 0; padding: 40px; }}
        .container {{ max-width: 1400px; margin: auto; }}
        header {{ margin-bottom: 30px; }}
        h1 {{ color: #1a1a1a; }}
        table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #edf0f2; }}
        th {{ background: #2c3e50; color: white; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 1px; position: sticky; top: 0; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover {{ background: #f8fafd; }}
        .user-cell {{ min-width: 200px; }}
        .user-name {{ font-weight: bold; color: #2c3e50; margin-bottom: 5px; }}
        .user-text {{ font-size: 12px; color: #7f8c8d; line-height: 1.4; }}
        audio {{ width: 220px; height: 35px; border-radius: 4px; }}
        .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }}
        .badge-ref {{ background: #e8f4fd; color: #2980b9; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>TTS Generation Benchmark</h1>
            <p>Comparing <strong>{len(versions)}</strong> models across <strong>{len(results)}</strong> test cases.</p>
        </header>
        <table>
            <thead>
                <tr>
                    <th class="user-cell">Character / Text</th>
                    <th>Reference <span class="badge badge-ref">REF</span></th>
                    {"".join([f"<th>{v}</th>" for v in versions])}
                </tr>
            </thead>
            <tbody>
"""
        for r in results:
            # We assume audio paths are relative to the report location or absolute but accessible
            # Here we use relative paths for better portability
            ref_rel = os.path.relpath(r['wav_path'], self.output_dir) if os.path.isabs(r['wav_path']) else r['wav_path']
            
            html_content += f"""
                <tr>
                    <td class="user-cell">
                        <div class="user-name">{r['name']}</div>
                        <div class="user-text">{r['text']}</div>
                    </td>
                    <td>
                        <audio controls src="{ref_rel}"></audio>
                    </td>
"""
            for v in versions:
                audio_path = r.get(v, "")
                if audio_path and os.path.exists(audio_path):
                    audio_rel = os.path.relpath(audio_path, self.output_dir)
                    html_content += f'<td><audio controls src="{audio_rel}"></audio></td>'
                else:
                    html_content += '<td><span style="color:#e74c3c; font-size:12px;">Missing</span></td>'
            
            html_content += "</tr>"

        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        report_path = os.path.join(self.output_dir, "index.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return report_path
