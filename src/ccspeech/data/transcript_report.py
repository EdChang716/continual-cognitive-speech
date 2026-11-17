"""Render a private, self-contained transcript audit without external assets."""
from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path

CONDITIONS = {'raw': '原始錄音 ASR', 'candidate': '候選受試者音訊 ASR', 'window': '保留視窗 ASR（已去除重疊）'}


def _text(value) -> str:
    return escape(str(value), quote=True)


def _rate(value) -> str:
    if value is None:
        return '—'
    number = float(value)
    return f'{100 * number:.2f}%' if math.isfinite(number) else '—'


def _scope(language: str, verified: bool) -> str:
    return '英文範圍已確認' if language == 'en' and verified else (
        '英文範圍待確認' if language == 'en' else '中文來源參考')


def _alignment(operations: list[dict], language: str) -> str:
    separator = ' ' if language == 'en' else ''
    pieces = []
    for operation in operations:
        reference = _text(separator.join(operation['reference_tokens']))
        hypothesis = _text(separator.join(operation['hypothesis_tokens']))
        tag = operation['tag']
        if tag == 'equal':
            pieces.append(f'<span class="equal">{reference}</span>')
        elif tag == 'replace':
            pieces.append(f'<span class="replace" title="S：替換"><del>{reference}</del><ins>{hypothesis}</ins></span>')
        elif tag == 'delete':
            pieces.append(f'<del class="delete" title="D：參考有、ASR 沒有">{reference}</del>')
        elif tag == 'insert':
            pieces.append(f'<ins class="insert" title="I：ASR 多出的內容">{hypothesis}</ins>')
        else:
            raise ValueError(f'Unknown alignment operation: {tag}')
    return separator.join(pieces) or '<span class="muted">沒有可顯示的文字單位</span>'


def _summary_rows(summary: dict) -> str:
    rows = []
    for group in summary['groups']:
        scope = _scope(group['language'], group['reference_scope_verified'])
        values = ''.join(f'<td>{_rate(group["conditions"][name]["error_rate"])}</td>' for name in CONDITIONS)
        rows.append(f'<tr><td>{_text(scope)}</td><td>{_text(group["split"])}</td>'
                    f'<td>{int(group["recordings"])}</td><td>{int(group["participants"])}</td>'
                    f'<td>{int(group["reference_units"]):,}</td>{values}</tr>')
    return ''.join(rows)


def _record_panel(row: dict) -> str:
    language = row['language']
    scope = 'verified' if row['reference_scope_verified'] else 'unverified'
    badges = ' · '.join(_text(value) for value in (language, row['split'], _scope(language, row['reference_scope_verified'])))
    sections = []
    for name, title in CONDITIONS.items():
        counts = row['counts'][name]
        scores = f'{_text(counts["metric"])} {_rate(counts["error_rate"])} · S {int(counts["S"])} / D {int(counts["D"])} / I {int(counts["I"])}'
        sections.append(f'<section class="condition"><h4>{title}</h4><div class="scores">{scores}</div>'
                        f'<details class="asr"><summary>原始 ASR 文字</summary><p class="transcript">{_text(row[name + "_asr"])}</p></details>'
                        f'<div class="alignment" aria-label="正規化文字對齊差異">{_alignment(row["alignments"][name], language)}</div></section>')
    flags = ', '.join(str(flag) for flag in row.get('quality_flags', [])) or '無額外旗標'
    source_words, covered_words = int(row['source_word_count']), int(row['window_covered_word_count'])
    return (f'<details class="record" data-language="{_text(language)}" data-split="{_text(row["split"])}" '
            f'data-scope="{scope}" data-recording="{_text(row["recording_id"])}">'
            f'<summary><strong>{_text(row["recording_id"])}</strong><span class="badges">{badges}</span></summary>'
            f'<div class="record-body"><p class="metadata">來源：{_text(row["source_dataset"])} · '
            f'參考範圍：{_text(row["reference_scope"])} · 視窗 {int(row["retained_windows"])} 段 · '
            f'覆蓋候選 ASR 詞項 {covered_words:,} / {source_words:,}（詞項不是 CER 字元單位）</p>'
            f'<div class="reference"><h3>原始受試者參考逐字稿</h3><p class="transcript">{_text(row["reference"])}</p></div>'
            f'<div class="conditions">{"".join(sections)}</div><p class="flags">品質旗標：{_text(flags)}</p></div></details>')


STYLE = '''
:root{color-scheme:light;font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;color:#23333d;background:#f3f6f6}
*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:32px 24px 64px}h1{font-size:30px;margin:6px 0 12px}h2{font-size:21px;margin:0 0 14px}h3,h4{margin:0 0 12px}h4{font-size:16px}.eyebrow{color:#146c66;font-weight:700;font-size:13px;letter-spacing:.1em}.lead{max-width:900px;line-height:1.8}.stats{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}.stat{background:white;border:1px solid #dae4e5;border-radius:10px;padding:12px 20px}.stat b{font-size:25px;display:block;color:#17615c}.card{background:white;border:1px solid #dae4e5;border-radius:12px;padding:22px;margin:20px 0}.notice{border-left:4px solid #d7a942;background:#fff9e9;padding:14px 18px;line-height:1.8}.muted,.metadata,.flags{color:#60717c;font-size:13px;line-height:1.7}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th{text-align:left;background:#edf5f4;color:#275d58}th,td{padding:10px 12px;border-bottom:1px solid #e3e9e9;white-space:nowrap}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}.controls label{display:grid;gap:5px;font-size:13px}.controls input,.controls select,button{padding:9px 12px;border:1px solid #b6c8c8;border-radius:7px;background:white;font:inherit}button{cursor:pointer;color:#175f5a}.legend{display:flex;gap:15px;flex-wrap:wrap;margin:16px 0;font-size:13px}.record{background:white;border:1px solid #d5e0e1;border-radius:9px;margin:10px 0}.record>summary{cursor:pointer;padding:15px 18px}.badges{font-size:12px;color:#61727e;margin-left:14px}.record-body{border-top:1px solid #e0e8e8;padding:16px 18px}.reference{background:#f0f6f5;border-radius:8px;padding:16px;margin:14px 0}.reference h3{font-size:15px}.transcript{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8;margin:0}.conditions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.condition{border:1px solid #dce5e5;border-radius:8px;padding:15px}.scores{font-size:12px;color:#506976;margin-bottom:12px}.asr{font-size:13px;margin-bottom:12px}.asr summary{cursor:pointer;color:#1f736c;margin-bottom:8px}.alignment{line-height:2.1;overflow-wrap:anywhere;font-size:14px}.delete,del{background:#fce1e1;color:#9a2626;text-decoration:line-through}.insert,ins{background:#dcefe4;color:#195d3b;text-decoration:underline}.replace{border:1px solid #ddc38c;border-radius:4px;padding:1px 2px;background:#fff7df}.replace ins{margin-left:4px}.flags{margin:14px 0 0;overflow-wrap:anywhere}.notes{line-height:1.8;font-size:13px}footer{color:#60717c;font-size:12px;margin-top:24px}[hidden]{display:none!important}@media(max-width:950px){.conditions{grid-template-columns:1fr}.badges{display:block;margin:7px 0 0}main{padding:20px 12px}}@media print{.controls,button{display:none}.record,.condition{break-inside:avoid}body{background:white}}
'''
SCRIPT = '''
const panels=[...document.querySelectorAll('.record')];
const controls=['language','split','scope','query'].map(id=>document.getElementById(id));
function filter(){const [language,split,scope,query]=controls.map(el=>el.value.trim().toLowerCase());let shown=0;for(const panel of panels){const d=panel.dataset;const keep=(!language||d.language===language)&&(!split||d.split===split)&&(!scope||d.scope===scope)&&(!query||d.recording.toLowerCase().includes(query));panel.hidden=!keep;if(keep)shown++;}document.getElementById('shown').textContent=shown+' / '+panels.length+' 段錄音';}
controls.forEach(el=>el.addEventListener('input',filter));
document.getElementById('expand').addEventListener('click',()=>panels.filter(p=>!p.hidden).forEach(p=>p.open=true));
document.getElementById('collapse').addEventListener('click',()=>panels.forEach(p=>p.open=false));filter();
'''


def render(report_dir: Path, output: Path | None = None) -> Path:
    summary = json.loads((report_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('schema_version') != 1:
        raise ValueError('Unsupported transcript comparison schema')
    rows = [json.loads(line) for line in (report_dir / 'differences.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != summary['recordings'] or len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('Summary/detail recording coverage differs or contains duplicates')
    order = {'train': 0, 'validation': 1, 'test': 2}
    rows.sort(key=lambda r: (r['language'], order.get(r['split'], 3), r['recording_id']))
    notes = ''.join(f'<li>{_text(note)}</li>' for note in summary.get('notes', []))
    panels = ''.join(_record_panel(row) for row in rows)
    document = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'"><title>TAUKADIAL 逐字稿比對 · 私人檢查報告</title><style>{STYLE}</style></head><body><main>
<header><div class="eyebrow">PRIVATE · TRANSCRIPT AUDIT</div><h1>TAUKADIAL 逐字稿比對</h1><p class="lead">對照受試者來源逐字稿、原始錄音 ASR、候選受試者音訊 ASR，以及訓練視窗涵蓋的 ASR 文字。英文以詞錯誤率 WER、中文以字錯誤率 CER 計算；數值越低，文字差異越少。</p></header>
<div class="stats"><div class="stat"><b>{int(summary['recordings']):,}</b>錄音</div><div class="stat"><b>{int(summary['participants']):,}</b>受試者</div><div class="stat"><b>3</b>文字條件</div></div>
<div class="notice"><strong>文字差異不等於講者辨識正確率。</strong>多出的字可能是訪談者內容或 ASR 錯誤；缺字可能來自音訊刪除或 ASR 漏辨。視窗條件另包含短錄音／尾段省略，已按來源詞項去除重疊，不能把所有差異都歸因於辨識品質。英文參考範圍待確認的錄音需分開解讀。</div>
<section class="card"><h2>整體與 split 比較</h2><p class="muted">每列以參考文字單位數加權：Σ(S + D + I) / Σ(reference units)。all 為該語言與參考範圍組的合計，非額外樣本。</p><div class="scroll"><table><thead><tr><th>參考組別</th><th>Split</th><th>錄音</th><th>受試者</th><th>參考單位</th><th>原始錄音</th><th>候選音訊</th><th>保留視窗</th></tr></thead><tbody>{_summary_rows(summary)}</tbody></table></div></section>
<section class="card"><h2>逐段檢查</h2><div class="controls"><label>語言<select id="language"><option value="">全部</option><option value="en">英文</option><option value="zh">中文</option></select></label><label>Split<select id="split"><option value="">全部</option><option>train</option><option>validation</option><option>test</option></select></label><label>參考範圍<select id="scope"><option value="">全部</option><option value="verified">已確認</option><option value="unverified">待確認</option></select></label><label>錄音 ID<input id="query" placeholder="輸入錄音 ID" autocomplete="off"></label><button id="expand" type="button">展開篩選結果</button><button id="collapse" type="button">全部收合</button></div><div class="legend"><span><span class="replace">S：替換</span></span><span><del>D：參考文字缺失</del></span><span><ins>I：ASR 額外文字</ins></span></div><p class="muted">差異區顯示正規化後的詞／字；原始文字另列。刪除項沒有可靠 ASR 時間戳，本頁不推定其音訊邊界。以下篩選只影響錄音面板。</p><p id="shown" aria-live="polite"></p></section>
<div id="recordings">{panels}</div><section class="card"><h2>比較說明</h2><ul class="notes">{notes}</ul></section><footer>私人資料檢查檔；包含受試者逐字稿，請勿公開。此頁不載入外部資源、不發出網路請求。</footer></main><script>{SCRIPT}</script></body></html>'''
    destination = output or report_dir / 'review.html'
    temporary = destination.with_suffix(destination.suffix + '.part')
    if destination.is_symlink() or temporary.is_symlink():
        raise ValueError('Refusing symlink report output')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(document, encoding='utf-8')
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('data/reports/transcript_comparison_v1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(render(args.report_dir, args.output))


if __name__ == '__main__':
    main()
