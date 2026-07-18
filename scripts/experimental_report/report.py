"""Markdown report generation for Regicide agent comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_report(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    report_config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Render a self-contained Italian experimental report."""
    destination = Path(output_dir)
    report_path = destination / "experimental_report.md"
    protocol = report_config["protocol"]
    content = "\n\n".join(
        [
            "# Regicide — report sperimentale",
            "",
            _protocol_section(games, protocol),
            _algorithm_section(games, report_config["agents"]),
            _results_section(summary),
            _statistics_section(pairwise, protocol),
            _visualization_section(),
            _interpretation_section(summary, pairwise, protocol),
            _limitations_section(),
        ]
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def write_tables(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Persist complete CSV tables and compact Markdown/LaTeX tables."""
    destination = Path(output_dir)
    summary_csv = destination / "summary.csv"
    pairwise_csv = destination / "pairwise_tests.csv"
    summary.to_csv(summary_csv, index=False)
    pairwise.to_csv(pairwise_csv, index=False)
    compact_summary = _compact_summary(summary)
    compact_pairs = _compact_pairwise(pairwise)
    markdown_path = destination / "tables.md"
    markdown_path.write_text(
        "## Risultati aggregati\n\n"
        + _as_markdown(compact_summary)
        + "\n\n## Confronti statistici appaiati\n\n"
        + _as_markdown(compact_pairs),
        encoding="utf-8",
    )
    latex_path = destination / "tables.tex"
    latex_path.write_text(
        compact_summary.to_latex(index=False, float_format="%.3f")
        + "\n\n"
        + compact_pairs.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )
    return [summary_csv, pairwise_csv, markdown_path, latex_path]


def _protocol_section(games: pd.DataFrame, protocol: dict[str, Any]) -> str:
    agents = games["agent"].nunique()
    seeds = games["seed"].nunique()
    return f"""## Protocollo

Il confronto usa {seeds} seed identici per ciascuno dei {agents} agenti
({len(games)} partite totali). L'abbinamento per seed riduce la variabilità
dovuta all'ordine iniziale delle carte. Ogni partita è limitata a
{protocol["max_decisions_per_game"]} decisioni. Tempi e metriche sono misurati
per singola partita; le run raw sono conservate in `datasets/games.csv`.

Intervalli di confidenza: {protocol["confidence_level"]:.0%}; campioni bootstrap:
{protocol["bootstrap_samples"]}."""


def _algorithm_section(
    games: pd.DataFrame,
    agent_specs: dict[str, dict[str, Any]],
) -> str:
    descriptions = []
    for agent_key in games["agent"].drop_duplicates():
        specification = agent_specs[agent_key]
        kwargs = specification.get("kwargs", {})
        parameters = ", ".join(
            f"`{key}={value}`" for key, value in kwargs.items() if key != "name"
        )
        suffix = f" Parametri: {parameters}." if parameters else ""
        descriptions.append(
            f"- **{specification['label']}** ({specification.get('family', 'N/D')}): "
            f"{specification['description']}{suffix}"
        )
    return "## Algoritmi confrontati\n\n" + "\n".join(descriptions)


def _results_section(summary: pd.DataFrame) -> str:
    table = _as_markdown(_compact_summary(summary))
    return f"""## Risultati

{table}

Le tabelle complete, comprensive di deviazione standard, mediana e intervalli
di confidenza per tutte le metriche, sono disponibili in `summary.csv` e
`tables.tex`."""


def _statistics_section(
    pairwise: pd.DataFrame,
    protocol: dict[str, Any],
) -> str:
    confidence = protocol["confidence_level"]
    if pairwise.empty:
        table = "È stato valutato un solo agente: i test appaiati non sono applicabili."
    else:
        table = _as_markdown(_compact_pairwise(pairwise))
    return f"""## Analisi statistiche

{table}

La percentuale di vittorie usa l'intervallo di Wilson. Le metriche continue
usano intervalli bootstrap al {confidence:.0%}. I confronti sulla vittoria
usano il test esatto di McNemar; quelli sui boss sconfitti il test di Wilcoxon
appaiato. I p-value sono corretti con Holm separatamente per ciascuna famiglia
di test. `Cohen dz` quantifica l'effetto appaiato sui boss; la probabilità di
superiorità assegna metà peso ai pareggi."""


def _visualization_section() -> str:
    return """## Grafici

![Dashboard complessiva](comprehensive_dashboard.png)

![Percentuale di vittorie](win_rate.png)

![Boss sconfitti](bosses_defeated.png)

![Tempo di esecuzione](execution_time.png)

![Trade-off qualità-costo](quality_cost_tradeoff.png)"""


def _interpretation_section(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    protocol: dict[str, Any],
) -> str:
    best_win = summary.sort_values("win_rate", ascending=False).iloc[0]
    best_progress = summary.sort_values("bosses_mean", ascending=False).iloc[0]
    fastest = summary.sort_values("duration_seconds_mean").iloc[0]
    alpha = 1.0 - float(protocol["confidence_level"])
    significant = _significant_comparisons(pairwise, alpha)
    return f"""## Lettura dei risultati

- Migliore percentuale di vittorie osservata: **{best_win['label']}**
  ({best_win['win_rate']:.1%}).
- Maggiore avanzamento medio: **{best_progress['label']}**
  ({best_progress['bosses_mean']:.2f} boss su 12).
- Minore tempo medio per partita: **{fastest['label']}**
  ({fastest['duration_seconds_mean']:.4f} s).
- Confronti significativi dopo correzione di Holm: {significant}.

Queste classifiche descrivono il campione; per concludere che un metodo sia
superiore occorre considerare congiuntamente intervalli, p-value, dimensione
dell'effetto e costo computazionale."""


def _limitations_section() -> str:
    return """## Limiti e riproducibilità

Il protocollo confronta la modalità solitario sulla stessa macchina. I tempi
non vanno confrontati direttamente con run ottenute su hardware o carichi
diversi. Un seed condiviso rende uguale lo stato iniziale, ma gli algoritmi
stocastici possono consumare numeri casuali in modo differente durante la
partita. Le politiche neurali disabilitate in configurazione non sono incluse
finché non viene indicato un checkpoint valido.

La configurazione effettiva e i dati per partita sono salvati insieme al report,
così tabelle e grafici possono essere rigenerati senza ripetere le simulazioni."""


def _compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    compact = summary[
        [
            "label",
            "n_games",
            "win_rate",
            "win_ci_low",
            "win_ci_high",
            "bosses_mean",
            "bosses_ci_low",
            "bosses_ci_high",
            "duration_seconds_mean",
            "decision_ms_mean",
        ]
    ].copy()
    compact.columns = [
        "Agente",
        "N",
        "Win rate",
        "Win CI low",
        "Win CI high",
        "Boss medi",
        "Boss CI low",
        "Boss CI high",
        "Tempo medio (s)",
        "Decisione media (ms)",
    ]
    return compact.round(4)


def _compact_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pairwise
    compact = pairwise[
        [
            "agent_a",
            "agent_b",
            "n_pairs",
            "win_rate_difference",
            "win_p_holm",
            "bosses_mean_difference",
            "bosses_p_holm",
            "bosses_cohen_dz",
            "median_duration_ratio",
        ]
    ].copy()
    compact.columns = [
        "Agente A",
        "Agente B",
        "N",
        "Δ win rate",
        "p Holm (win)",
        "Δ boss",
        "p Holm (boss)",
        "Cohen dz",
        "Rapporto tempi A/B",
    ]
    return compact.round(4)


def _as_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "Non applicabile."
    headers = [str(column) for column in table.columns]
    rows = [[_markdown_cell(value) for value in row] for row in table.itertuples(False)]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *data_lines])


def _markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _significant_comparisons(pairwise: pd.DataFrame, alpha: float) -> str:
    if pairwise.empty:
        return "non applicabile"
    win_count = int((pairwise["win_p_holm"] < alpha).sum())
    boss_count = int((pairwise["bosses_p_holm"] < alpha).sum())
    return f"{win_count} sulla vittoria e {boss_count} sui boss (α={alpha:.3f})"
