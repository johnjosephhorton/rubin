"""
Insert the two matched-specification comparison figures into the report.

The report exists in two builds and they need the figures delivered differently:

  report_source.html  the Artifact build. The Artifact CSP blocks external images, so the
                      figures must be inlined as base64 data URIs. They are downscaled and
                      palette-quantised first, which takes the pair from about 1 MB to about
                      250 KB with no visible loss on line art.
  report_print.html   the local build that build_report_pdf.sh feeds to headless Chrome over
                      file://, which can read the PNGs directly. This one references them by
                      relative path, so the PDF gets full resolution and the committed HTML
                      stays small.

Idempotent: a figure block already present is replaced rather than duplicated, so this can be
re-run after either the report or the figures change.

Run figure_SAD_matched.py and figure_SAE_matched.py first, then this, then build_report_pdf.sh.
"""
import base64
import io
import os
import re

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PLOTS = os.path.join(REPO, "writeup", "plots", "efi_matched_exposure")

EMBED_WIDTH = 1900   # px, plenty for a page at 180 mm
EMBED_COLORS = 128   # palette size; line art quantises cleanly

FIGURES = [
    dict(
        key="sad",
        png="fragmentation_index_robustness_definition_1_matched.png",
        anchor="The exposure half of SA.D gets stronger, not weaker.</p>",
        caption=("Prompt robustness under the matched specification. Top row the AI exposure "
                 "coefficient, bottom row the fragmentation index, across eleven GPT task "
                 "orderings with 90% intervals; the main prompt is red and the dashed line is "
                 "the cross-prompt mean. The fragmentation row sits on zero in all three "
                 "columns while exposure sits well clear of it. The published counterpart is "
                 "the table above; a twelve-panel side-by-side version is at "
                 "writeup/plots/efi_matched_exposure/"
                 "fragmentation_index_robustness_definition_1_comparison.png."),
    ),
    dict(
        key="sae",
        png="frag_logic_threshold_heatmap_def1_comparison.png",
        anchor="The sign pattern goes from 33 of 39 negative to 16 of 39, which is a coin flip.</p>",
        caption=("Frequency cuts, published against matched. Each row is one frequency rule, "
                 "each panel one fixed-effects specification. Navy is the published "
                 "specification, orange the matched one, and the grey arrow is the shift "
                 "between them. Bars are 95% intervals. The unpruned row is shaded. Every "
                 "matched interval crosses zero."),
    ),
]

FIG_CSS = """
<style id="figure-css">
figure.plate{margin:26px 0 8px;padding:0}
figure.plate .mount{
  background:#fff;border:1px solid var(--rule);border-radius:3px;
  padding:10px;overflow-x:auto;
}
figure.plate img{display:block;width:100%;height:auto;min-width:520px}
figure.plate figcaption{
  font-size:12.5px;color:var(--muted);margin-top:9px;max-width:72ch;line-height:1.45;
}
@media print{
  figure.plate{break-inside:avoid;margin:20px 0 6px}
  figure.plate .mount{overflow:visible;padding:4px;border-color:#d8dbd6}
  figure.plate img{min-width:0}
  figure.plate figcaption{font-size:11px;margin-top:7px}
}
</style>
"""


def block(key, src, caption):
    return (f'\n  <figure class="plate" id="fig-{key}">\n'
            f'    <div class="mount"><img alt="{caption[:60]}" src="{src}"></div>\n'
            f'    <figcaption>{caption}</figcaption>\n'
            f'  </figure>')


def data_uri(path):
    im = Image.open(path).convert("RGB")
    w = min(EMBED_WIDTH, im.size[0])
    im = im.resize((w, round(im.size[1] * w / im.size[0])), Image.LANCZOS)
    im = im.quantize(colors=EMBED_COLORS, method=Image.MEDIANCUT)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    raw = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), len(raw)


def apply(path, src_for):
    with open(path, encoding="utf-8") as f:
        html = f.read()
    # drop any previous blocks and css so the step is idempotent
    html = re.sub(r'\n  <figure class="plate" id="fig-[a-z]+">.*?</figure>', "", html, flags=re.S)
    html = re.sub(r'<style id="figure-css">.*?</style>\n?', "", html, flags=re.S)

    for fig in FIGURES:
        if fig["anchor"] not in html:
            raise SystemExit(f"anchor not found in {os.path.basename(path)}: {fig['anchor'][:50]}")
        html = html.replace(fig["anchor"],
                            fig["anchor"] + block(fig["key"], src_for(fig), fig["caption"]))

    # css goes right after the last existing </style> so it can use the theme tokens
    i = html.rindex("</style>") + len("</style>")
    html = html[:i] + "\n" + FIG_CSS.strip() + html[i:]

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return len(html)


if __name__ == "__main__":
    for fig in FIGURES:
        p = os.path.join(PLOTS, fig["png"])
        if not os.path.exists(p):
            raise SystemExit(f"missing figure {p}\nrun figure_SAD_matched.py and "
                             f"figure_SAE_matched.py first")

    uris = {}
    for fig in FIGURES:
        uri, nbytes = data_uri(os.path.join(PLOTS, fig["png"]))
        uris[fig["key"]] = uri
        print(f"  {fig['png'][:52]:52} embedded at {EMBED_WIDTH}px, {nbytes // 1024} KB")

    n = apply(os.path.join(_HERE, "report_source.html"), lambda f: uris[f["key"]])
    print(f"report_source.html  {n // 1024} KB  (figures inlined as data URIs, for the Artifact)")

    n = apply(os.path.join(_HERE, "report_print.html"),
              lambda f: f"../../writeup/plots/efi_matched_exposure/{f['png']}")
    print(f"report_print.html   {n // 1024} KB  (figures by relative path, full resolution)")
    print("\nnow run build_report_pdf.sh")
