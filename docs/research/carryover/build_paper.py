"""Build the carryover manuscript from audited empirical evidence only."""

import argparse
import hashlib
import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
TEXT = json.loads((HERE / "paper_text.json").read_text())
INK = colors.HexColor("#17354a")
BLUE = colors.HexColor("#276b9a")


def _t(key, **values):
    return TEXT[key].format(**values)


def _verify_audit_binding(result_bytes, frozen_bytes, completion):
    required = {"source", "likelihood", "predictions", "paired"}
    if (
        not isinstance(completion, dict)
        or completion.get("status") != "complete_verified"
        or completion.get("primary_verified") is not True
        or set(completion.get("verification_stages", {})) != required
        or not all(v is True for v in completion["verification_stages"].values())
    ):
        raise ValueError("The audit has no complete executed verification record.")
    expected = {
        "audited_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "audited_frozen_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(
                "The executed audit does not cover these publication inputs."
            )


def _number(value, digits=4):
    if value is None:
        return "not identified"
    value = float(value)
    return f"{value:.2e}" if value and abs(value) < 0.001 else f"{value:.{digits}f}"


def _interval(value):
    return "[" + ", ".join(_number(v) for v in value) + "]"


def _font_directory(requested):
    if requested:
        return Path(requested)
    candidates = [
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype/liberation"),
    ]
    bundle = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
        / "native/libreoffice-headless"
    )
    candidates.extend(p.parent for p in bundle.glob("**/LiberationSans-Regular.ttf"))
    for directory in candidates:
        if (directory / "LiberationSans-Regular.ttf").is_file():
            return directory
    raise FileNotFoundError("Provide --font-dir containing LiberationSans font files.")


def _styles(font_dir):
    for name, file in (
        ("Body", "LiberationSans-Regular.ttf"),
        ("BodyBold", "LiberationSans-Bold.ttf"),
        ("BodyItalic", "LiberationSans-Italic.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily(
        "Body",
        normal="Body",
        bold="BodyBold",
        italic="BodyItalic",
        boldItalic="BodyBold",
    )
    return {
        "body": ParagraphStyle(
            "body",
            fontName="Body",
            fontSize=9.6,
            leading=13.1,
            spaceAfter=7,
            textColor=INK,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Body",
            fontSize=8,
            leading=10.4,
            spaceAfter=5,
            textColor=INK,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Body",
            fontSize=8.2,
            leading=10.7,
            spaceAfter=10,
            textColor=INK,
        ),
        "title": ParagraphStyle(
            "title",
            fontName="BodyBold",
            fontSize=20,
            leading=24,
            spaceAfter=12,
            textColor=INK,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="BodyBold",
            fontSize=14,
            leading=17,
            spaceBefore=6,
            spaceAfter=9,
            textColor=BLUE,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="BodyBold",
            fontSize=10.8,
            leading=14,
            spaceBefore=7,
            spaceAfter=6,
            textColor=BLUE,
        ),
    }


def _p(text, styles, kind="body"):
    return Paragraph(
        text.replace("–", "-").replace("—", "-").replace("−", "-"), styles[kind]
    )


def _table(rows, styles, widths=None):
    cells = [[_p(str(cell), styles, "small") for cell in row] for row in rows]
    table = Table(cells, colWidths=widths or [300, 205], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f5")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLUE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d5dfe5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _figure(name, caption, styles, width=505):
    file = HERE / "figures" / name
    w, h = ImageReader(str(file)).getSize()
    return KeepTogether(
        [
            Image(str(file), width=width, height=width * h / w),
            Spacer(1, 5),
            _p(caption, styles, "caption"),
        ]
    )


def _footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c3d2dc"))
    canvas.line(45, 36, A4[0] - 45, 36)
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(INK)
    canvas.drawString(
        45, 24, "HABS | Exploratory subjective-EEG carryover study | 9 November 2026"
    )
    canvas.drawRightString(A4[0] - 45, 24, str(document.page))
    canvas.restoreState()


def _build(file, story):
    document = SimpleDocTemplate(
        str(file),
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=39,
        bottomMargin=47,
        title="Subjective and EEG carryover across sessions",
        author="HABS research project",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def _conclusion(result):
    component = result["variance_components"]
    lo, hi = component["covariance_ci95"]
    if lo > 0 and component["null_bootstrap_p_two_sided"] < 0.05:
        association = _t("p001")
    elif hi < 0 and component["null_bootstrap_p_two_sided"] < 0.05:
        association = _t("p002")
    else:
        association = (
            "Positive cross-session repeatability of the personal contrast "
            "was not established."
        )
    gain = result["prediction"]["full_vs_ablation"]
    if gain["ci95"][0] > 0:
        prediction = (
            "The fixed personal correction improved the primary "
            "grouped J2 forecast comparison."
        )
    elif gain["ci95"][1] < 0:
        prediction = (
            "The fixed personal correction worsened the primary "
            "grouped J2 forecast comparison."
        )
    else:
        prediction = (
            "The primary grouped J2 comparison did not establish "
            "a predictive improvement."
        )
    return association, prediction


def _manuscript(result, frozen, precision, styles):
    m, c, p = result["structure"], result["variance_components"], result["prediction"]
    association, prediction = _conclusion(result)
    weak = c["correlation_weakly_identified"]
    rho_text = (
        _t("p003")
        if weak
        else (
            "The conditional correlation profile covered "
            f"{_interval(c['rho_profile95_grid'])}."
        )
    )
    story = [
        _p(
            "Do subjective and EEG carryover differences repeat across sessions?",
            styles,
            "title",
        ),
        _p(
            "A fixed-catalogue, internally validated analysis of HABS recordings",
            styles,
            "h2",
        ),
        _p(
            "Research report | 9 November 2026 | Exploratory secondary analysis",
            styles,
            "small",
        ),
        _p("Abstract", styles, "h2"),
        _p(
            _t(
                "p004",
                v0=m["people"],
                v1=m["transitions"],
                v2=_number(c["covariance"]),
                v3=_interval(c["covariance_ci95"]),
                v4=c["null_bootstrap_p_two_sided"],
                v5=_number(p["full_vs_ablation"]["gain"]),
                v6=_interval(p["full_vs_ablation"]["ci95"]),
                v7=association,
                v8=prediction,
            ),
            styles,
        ),
        _p("1. Motivation", styles, "h1"),
        _p(
            _t("p005"),
            styles,
        ),
        _p("2. Data and measurements", styles, "h1"),
        _p(
            _t("p006", v0=m["by_day"]["1"], v1=m["by_day"]["2"]),
            styles,
        ),
        _figure(
            "figure1_transition_support.png",
            _t("p007"),
            styles,
        ),
    ]
    story += [
        PageBreak(),
        _p("3. Estimand and uncertainty", styles, "h1"),
        _p(
            _t("p008"),
            styles,
        ),
        _p(
            _t("p009", v0=m["reference_gap_s"]),
            styles,
        ),
        _p(
            _t("p010"),
            styles,
        ),
        _p(
            _t("p011"),
            styles,
        ),
        _p("4. Identifiability and pre-effect precision", styles, "h1"),
        _p(
            _t("p012"),
            styles,
        ),
        _figure(
            "figure2_simulated_precision.png",
            _t("p013"),
            styles,
        ),
    ]
    story += [
        PageBreak(),
        _p("5. Cross-session contrast results", styles, "h1"),
        _table(
            [
                ["Quantity", "Estimate"],
                [
                    "Paired identifiers / transitions",
                    f"{m['people']} / {m['transitions']:,}",
                ],
                ["Cross-day random-effect covariance", _number(c["covariance"])],
                ["Model-based 95% bootstrap interval", _interval(c["covariance_ci95"])],
                [
                    "Two-sided null-bootstrap p",
                    f"{c['null_bootstrap_p_two_sided']:.3f}",
                ],
                [
                    "Between-person SD, J1 / J2",
                    " / ".join(_number(v) for v in c["random_effect_sd"]),
                ],
                [
                    "Approximate correlation profile range",
                    _interval(c["rho_profile95_grid"]),
                ],
            ],
            styles,
        ),
        Spacer(1, 8),
        _p(
            association + " " + rho_text + _t("p014"),
            styles,
        ),
        _figure(
            "figure3_personal_contrasts.png",
            _t("p015"),
            styles,
        ),
        _figure(
            "figure4_reproducibility_uncertainty.png",
            _t("p016"),
            styles,
        ),
    ]
    story += [
        PageBreak(),
        _p("6. Predictive corroboration on J2", styles, "h1"),
        _p(
            _t("p017"),
            styles,
        ),
        _table(
            [["Frozen predictor", "Equal-person MAE"]]
            + [
                [
                    {
                        "ablation": "Exact personal-slope ablation",
                        "full": "Full personal carryover correction",
                        "refit": "Refitted restricted calibration",
                    }[k],
                    _number(p["models"][k]["participant_macro_mae"]),
                ]
                for k in ("ablation", "full", "refit")
            ],
            styles,
        ),
        Spacer(1, 7),
        _p(
            _t(
                "p018",
                v0=_number(p["full_vs_ablation"]["gain"]),
                v1=_interval(p["full_vs_ablation"]["ci95"]),
                v2=p["full_vs_ablation"]["people_helped"],
                v3=p["people"],
                v4=_number(p["full_vs_refitted_restriction"]["gain"]),
                v5=_interval(p["full_vs_refitted_restriction"]["ci95"]),
                v6=prediction,
                v7=p["trials"],
            ),
            styles,
        ),
        _figure(
            "figure5_personal_prediction_gain.png",
            _t("p019"),
            styles,
        ),
        _p("7. Interpretation", styles, "h1"),
        _p(
            _t("p020"),
            styles,
        ),
    ]
    story += [
        PageBreak(),
        _p("8. Limits and reproducibility", styles, "h1"),
        _p(
            _t("p021"),
            styles,
        ),
        _p(
            _t("p022"),
            styles,
        ),
        _p(
            _t("p023"),
            styles,
        ),
        _p("Reproducibility", styles, "h2"),
        _p(
            _t(
                "p024",
                v0=escape(frozen["source_revision"]),
                v1=escape(frozen["source_revision"][:12]),
            ),
            styles,
        ),
        _p("References", styles, "h1"),
    ]
    refs = [
        _t("p025"),
        _t("p026"),
        _t("p027"),
        _t("p028"),
        _t("p029"),
        _t("p030"),
        _t("p031"),
        _t("p032"),
    ]
    story.extend(_p(escape(r), styles, "small") for r in refs)
    return story


def _supplement(result, frozen, precision, feasibility, styles):
    m = result["structure"]
    story = [
        _p("Supporting evidence", styles, "title"),
        _p("S1. Eligibility and fixed model specification", styles, "h1"),
        _table(
            [["Stage / exclusion reason", "Count"]]
            + [[escape(k), str(v)] for k, v in feasibility["pair_reasons"].items()]
            + [
                ["Final paired identifiers", str(m["people"])],
                [
                    "Final J1 / J2 transitions",
                    f"{m['by_day']['1']} / {m['by_day']['2']}",
                ],
            ],
            styles,
        ),
        Spacer(1, 9),
        _p(
            _t("p033"),
            styles,
        ),
        _table(
            [["Frozen setting", "Value"]]
            + [
                [escape(k), escape(f"{v:g}" if isinstance(v, float) else str(v))]
                for k, v in frozen["settings"].items()
            ],
            styles,
        ),
        Spacer(1, 8),
        _p(
            _t("p034"),
            styles,
        ),
        _p(
            _t(
                "p035",
                v0=escape(frozen["source_revision"]),
                v1=escape(frozen["frozen_at_utc"]),
                v2=escape(frozen["scientific_code_sha256"]),
            ),
            styles,
            "small",
        ),
        PageBreak(),
        _p("S2. Prespecified sensitivity analyses", styles, "h1"),
    ]
    rows = [["Specification", "IDs / pairs", "Covariance", "Correlation profile"]]
    for name, item in result["sensitivities"].items():
        if item["status"] == "completed":
            rows.append(
                [
                    escape(name.replace("_", " ")),
                    f"{item['people']} / {item['transitions']}",
                    _number(item["covariance"]),
                    _interval(item["rho_profile95_grid"]),
                ]
            )
        else:
            rows.append(
                [
                    escape(name.replace("_", " ")),
                    f"{item.get('people', '-')} / {item.get('transitions', '-')}",
                    "-",
                    escape(item["status"].replace("_", " ")),
                ]
            )
    story += [
        _table(rows, styles, [154, 80, 105, 166]),
        Spacer(1, 10),
        _p(
            _t("p036"),
            styles,
        ),
        _p("S3. Model-based uncertainty and predictive uncertainty", styles, "h1"),
        _p(
            _t("p037"),
            styles,
        ),
        _p("Saving and reading the forecasts", styles, "h2"),
        _p(
            _t("p038"),
            styles,
        ),
        PageBreak(),
        _p("S4. Pre-effect synthetic precision diagnostics", styles, "h1"),
        _p(
            _t("p039"),
            styles,
        ),
    ]
    rows = [["Scenario / SD", "True rho", "Estimated covariance: 2.5%, median, 97.5%"]]
    for s in precision["scenarios"]:
        rows.append(
            [
                (
                    "Shared scale only"
                    if s["scenario"] == "shared_scale_only"
                    else f"Random SD {s['true_random_sd']:.2f}"
                ),
                "undefined"
                if s["true_correlation"] is None
                else str(s["true_correlation"]),
                ", ".join(_number(v) for v in s["estimated_covariance_quantiles"]),
            ]
        )
    story += [
        _table(rows, styles, [145, 80, 280]),
        Spacer(1, 10),
        _p(
            _t("p040"),
            styles,
        ),
        _p("Evidence files", styles, "h2"),
        _p(
            _t("p041"),
            styles,
        ),
    ]
    return story


def main():
    """Create the manuscript and supplement after verifying the empirical audit.

    Returns
    -------
    None
        Two final PDFs are written beside this source script.

    Raises
    ------
    ValueError
        If the source evidence is not a completed, audited empirical study.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-dir")
    args = parser.parse_args()
    result = json.loads((HERE / "results/temporal_results.json").read_text())
    frozen_path = HERE / "results/temporal_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    audit = (HERE / "results/audit.md").read_text()
    if result["status"] != "empirical_carryover_complete" or "**error:**" in audit:
        raise ValueError(
            "PDF authoring requires a completed successful empirical audit."
        )
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != result["frozen_sha256"]:
        raise ValueError("The empirical freeze differs from the audited evidence.")
    _verify_audit_binding(
        (HERE / "results/temporal_results.json").read_bytes(),
        frozen_path.read_bytes(),
        json.loads((HERE / "results/audit_completion.json").read_text()),
    )
    styles = _styles(_font_directory(args.font_dir))
    precision = json.loads((PROJECT / "data/temporal_precision.json").read_text())
    feasibility = json.loads((PROJECT / "data/temporal_feasibility.json").read_text())
    _build(HERE / "manuscript.pdf", _manuscript(result, frozen, precision, styles))
    _build(
        HERE / "supplement.pdf",
        _supplement(result, frozen, precision, feasibility, styles),
    )
    print("Created empirical carryover manuscript and supporting evidence.")


if __name__ == "__main__":
    main()
