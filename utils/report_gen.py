import json
import shutil
from html import escape
from pathlib import Path

from utils.artifacts import sha256_file


class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def _copy_asset(self, source, relative):
        destination = self.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return relative.as_posix(), {
            "path": relative.as_posix(),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }

    def generate_html(self, results, versions, reference_policy="copy"):
        assets = []
        rows = []
        for index, result in enumerate(results, start=1):
            row = {"name": str(result["name"]), "text": str(result["text"]), "audio": {}}
            if reference_policy == "copy":
                relative, record = self._copy_asset(
                    result["wav_path"], Path("assets") / "references" / f"reference-{index:04d}.wav"
                )
                row["audio"]["reference"] = relative
                assets.append(record)
            for version in versions:
                source = result.get(version)
                if source and Path(source).is_file():
                    relative, record = self._copy_asset(
                        source, Path("assets") / "generated" / version / f"sample-{index:04d}.wav"
                    )
                    row["audio"][version] = relative
                    assets.append(record)
            rows.append(row)

        html_content = self._render(rows, versions)
        report_path = self.output_dir / "index.html"
        report_path.write_text(html_content, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "reference_policy": reference_policy,
            "assets": assets,
            "index": {"path": "index.html", "sha256": sha256_file(report_path)},
        }
        (self.output_dir / "report_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(report_path)

    @staticmethod
    def _render(rows, versions):
        columns = "".join(f"<th>{escape(str(version))}</th>" for version in versions)
        body = []
        for row in rows:
            reference = row["audio"].get("reference")
            reference_cell = (
                f'<audio controls preload="none" src="{escape(reference, quote=True)}"></audio>'
                if reference
                else '<span class="missing">Omitted</span>'
            )
            cells = []
            for version in versions:
                source = row["audio"].get(version)
                cells.append(
                    f'<td><audio controls preload="none" src="{escape(source, quote=True)}"></audio></td>'
                    if source
                    else '<td><span class="missing">Missing</span></td>'
                )
            body.append(
                "<tr>"
                f'<td class="subject"><strong>{escape(row["name"])}</strong>'
                f'<span>{escape(row["text"])}</span></td>'
                f"<td>{reference_cell}</td>{''.join(cells)}</tr>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TTS Model Comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f4f5f7; color: #202124; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; overflow-x: auto; }}
    h1 {{ font-size: 24px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 12px; border: 1px solid #dfe1e5; text-align: left; }}
    th {{ background: #263238; color: white; position: sticky; top: 0; }}
    .subject {{ min-width: 180px; }}
    .subject span {{ display: block; margin-top: 4px; color: #5f6368; font-size: 13px; }}
    audio {{ width: 220px; }}
    .missing {{ color: #b3261e; }}
  </style>
</head>
<body><main>
  <h1>TTS Model Comparison</h1>
  <p>{len(rows)} test cases across {len(versions)} models.</p>
  <table><thead><tr><th>Character / Text</th><th>Reference</th>{columns}</tr></thead>
  <tbody>{''.join(body)}</tbody></table>
</main></body>
</html>
"""
