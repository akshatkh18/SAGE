import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from app.utils.logger import logger
from app.utils.exceptions import ReportGenerationError

REPORT_DIR = "artifacts/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def get_styles():
    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=colors.HexColor("#4F8BF9"),
            spaceAfter=6,
            alignment=TA_CENTER
        ),
        "subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#888888"),
            spaceAfter=20,
            alignment=TA_CENTER
        ),
        "h1": ParagraphStyle(
            "CustomH1",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#4F8BF9"),
            spaceBefore=16,
            spaceAfter=8
        ),
        "h2": ParagraphStyle(
            "CustomH2",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#333333"),
            spaceBefore=10,
            spaceAfter=6
        ),
        "body": ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#222222"),
            spaceAfter=6,
            leading=14
        ),
        "caption": ParagraphStyle(
            "CustomCaption",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#888888"),
            spaceAfter=4,
            alignment=TA_CENTER
        ),
    }
    return custom

def fig_to_image(fig, width=6*inch, height=3.5*inch):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image(buf, width=width, height=height)
    return img

def df_to_table(df: pd.DataFrame, col_widths=None):
    data = [list(df.columns)] + df.values.tolist()
    data = [[str(cell) for cell in row] for row in data]

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F8BF9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FF")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table

def generate_report(
    dataset_name: str,
    target_column: str,
    profile: dict,
    insights: list,
    preprocessing_report: list,
    dropped_columns: list,
    arena_result: dict,
    feature_importance_df: pd.DataFrame = None,
) -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORT_DIR, f"SAGE_Report_{timestamp}.pdf")

        doc = SimpleDocTemplate(
            report_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        styles = get_styles()
        story = []

        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("SAGE", styles["title"]))
        story.append(Paragraph("Smart Adaptive Guided Experimentation", styles["subtitle"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4F8BF9")))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Experiment Report", styles["h1"]))
        story.append(Spacer(1, 0.2*inch))

        meta_data = [
            ["Dataset", dataset_name],
            ["Target Column", target_column],
            ["Task Type", arena_result["task"].capitalize()],
            ["Best Model", arena_result["best_model_name"]],
            ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]
        meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4F8BF9")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEEEEE")),
        ]))
        story.append(meta_table)
        story.append(PageBreak())

        story.append(Paragraph("1. Dataset Overview", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            f"The dataset <b>{dataset_name}</b> contains <b>{profile['n_rows']:,} rows</b> "
            f"and <b>{profile['n_cols']} columns</b>, with "
            f"<b>{profile.get('duplicate_rows', profile.get('duplicated_rows', 0))} duplicate rows</b> detected.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph("Column Statistics", styles["h2"]))
        col_rows = []
        for col in profile["columns"]:
            row = [
                col["name"],
                col["dtype"],
                f"{col['missing_pct']}%",
                str(col["unique"]),
            ]
            if "skew" in col:
                row.append(str(col["skew"]))
                row.append(f"{col.get('outlier_pct', 'N/A')}%")
            else:
                row.append("N/A")
                row.append("N/A")
            col_rows.append(row)

        col_df = pd.DataFrame(
            col_rows,
            columns=["Column", "Type", "Missing %", "Unique", "Skew", "Outlier %"]
        )
        story.append(df_to_table(col_df))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("2. Automated Insights", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        story.append(Spacer(1, 0.1*inch))

        severity_colors = {
            "critical": colors.HexColor("#FF4444"),
            "warning": colors.HexColor("#FFA500"),
            "info": colors.HexColor("#4F8BF9")
        }

        for ins in insights:
            sev = ins.get("severity", "info")
            col_name = ins.get("column") or "General"
            msg = ins.get("message", "")
            action = ins.get("suggested_action", "")
            color = severity_colors.get(sev, colors.black)

            insight_data = [
                [f"[{sev.upper()}] {col_name}", msg],
                ["Action", action]
            ]
            ins_table = Table(insight_data, colWidths=[1.5*inch, 5.25*inch])
            ins_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (0, 0), color),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(ins_table)
            story.append(Spacer(1, 0.05*inch))

        story.append(PageBreak())

        story.append(Paragraph("3. Preprocessing Pipeline", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            f"SAGE applied <b>{len(preprocessing_report)} transformations</b> to prepare the data for modeling. "
            f"<b>{len(dropped_columns)} columns</b> were dropped: {', '.join(dropped_columns) if dropped_columns else 'None'}.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1*inch))

        prep_rows = [[r["column"], r["action"], r["reason"][:80] + "..." if len(r["reason"]) > 80 else r["reason"]]
                     for r in preprocessing_report]
        prep_df = pd.DataFrame(prep_rows, columns=["Column", "Action", "Reason"])
        story.append(df_to_table(prep_df, col_widths=[1.2*inch, 1.8*inch, 3.75*inch]))
        story.append(PageBreak())

        story.append(Paragraph("4. Model Arena Results", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            f"SAGE trained <b>{len(arena_result['leaderboard'])} models</b> and selected "
            f"<b>{arena_result['best_model_name']}</b> as the best performer based on "
            f"{'F1 score' if arena_result['task'] != 'regression' else 'R2 score'}.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph("Leaderboard", styles["h2"]))
        lb = arena_result["leaderboard"].copy()
        lb = lb.astype(str)
        story.append(df_to_table(lb))
        story.append(Spacer(1, 0.2*inch))

        primary = "F1" if arena_result["task"] != "regression" else "R2"
        leaderboard = arena_result["leaderboard"]
        if primary in leaderboard.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_list = ["#00cc66" if i == 0 else "#4F8BF9" for i in range(len(leaderboard))]
            ax.barh(leaderboard["Model"], leaderboard[primary], color=colors_list)
            ax.set_xlabel(primary)
            ax.set_title(f"Model Comparison by {primary}")
            ax.invert_yaxis()
            plt.tight_layout()
            story.append(fig_to_image(fig))
            story.append(Paragraph(f"Figure: Model comparison by {primary} score", styles["caption"]))
            plt.close()

        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Tuning Impact", styles["h2"]))
        untuned = arena_result["untuned_metrics"]
        tuned = arena_result["tuned_metrics"]
        tuning_rows = []
        for metric in untuned:
            before = untuned[metric]
            after = tuned.get(metric, "N/A")
            delta = round(float(after) - float(before), 4) if after != "N/A" else "N/A"
            tuning_rows.append([metric, str(before), str(after), f"+{delta}" if isinstance(delta, float) and delta > 0 else str(delta)])

        tuning_df = pd.DataFrame(tuning_rows, columns=["Metric", "Before Tuning", "After Tuning", "Delta"])
        story.append(df_to_table(tuning_df))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Best Hyperparameters", styles["h2"]))
        params_rows = [[k, str(round(float(v), 4)) if isinstance(v, float) else str(v)]
                       for k, v in arena_result["best_params"].items()]
        params_df = pd.DataFrame(params_rows, columns=["Parameter", "Value"])
        story.append(df_to_table(params_df, col_widths=[3*inch, 3.75*inch]))

        story.append(PageBreak())

        story.append(Paragraph("5. Feature Importance (SHAP)", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        story.append(Spacer(1, 0.1*inch))

        if feature_importance_df is not None:
            story.append(Paragraph(
                "Feature importance ranked by mean absolute SHAP value across all test samples.",
                styles["body"]
            ))
            story.append(Spacer(1, 0.1*inch))
            story.append(df_to_table(
                feature_importance_df.head(10),
                col_widths=[3.5*inch, 3.25*inch]
            ))
            story.append(Spacer(1, 0.2*inch))

            top10 = feature_importance_df.head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top10["Feature"], top10["Mean |SHAP|"], color="#4F8BF9")
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Top 10 Features by SHAP Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            story.append(fig_to_image(fig))
            story.append(Paragraph("Figure: Top 10 features by mean absolute SHAP value", styles["caption"]))
            plt.close()
        else:
            story.append(Paragraph(
                "SHAP analysis was not run in this session. Run the Explainability module to include feature importance in the report.",
                styles["body"]
            ))

        story.append(PageBreak())

        story.append(Paragraph("6. Summary", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        story.append(Spacer(1, 0.1*inch))

        primary_score = tuned.get("F1") or tuned.get("R2") or "N/A"
        auc_score = tuned.get("AUC", "N/A")

        story.append(Paragraph(
            f"SAGE successfully completed an end-to-end ML experiment on <b>{dataset_name}</b>. "
            f"After profiling {profile['n_rows']:,} rows, generating {len(insights)} automated insights, "
            f"and applying {len(preprocessing_report)} preprocessing transformations, "
            f"<b>{arena_result['best_model_name']}</b> was selected as the best model with a "
            f"{primary} score of <b>{primary_score}</b>"
            + (f" and AUC of <b>{auc_score}</b>" if auc_score != "N/A" else "") + ".",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            f"Model saved to: <b>{arena_result['model_path']}</b>",
            styles["body"]
        ))
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#4F8BF9")))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Generated by SAGE — Smart Adaptive Guided Experimentation",
            styles["caption"]
        ))

        doc.build(story)
        logger.info(f"Report generated: {report_path}")
        return report_path

    except ReportGenerationError:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise ReportGenerationError(f"Report generation failed: {e}") from e