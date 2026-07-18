# Regicide — protocollo del report sperimentale

La pipeline in `scripts/experimental_report/` confronta gli agenti implementati
con un protocollo riproducibile e genera automaticamente il materiale del
report: descrizione dei metodi, dati raw, tabelle, grafici e analisi
statistiche.

## Esecuzione completa

```bash
python -m scripts.experimental_report.orchestrator
```

La configurazione è nella sezione `experimental_report` di `config.yaml`.
È possibile sovrascrivere da CLI il campione, il seed e gli agenti:

```bash
python -m scripts.experimental_report.orchestrator \
  --agents random heuristic pimc ismcts \
  --games 100 \
  --base-seed 20260718
```

Random, Heuristic, PIMC e ISMCTS sono abilitati inizialmente. PPO e AlphaZero
sono già registrati, ma restano disabilitati finché i rispettivi checkpoint non
sono disponibili. Per includerli basta configurare il percorso del modello e
impostare `enabled: true`.

## Protocollo

Ogni agente gioca con la stessa sequenza di seed. Il confronto è quindi
appaiato: per ciascun seed i metodi partono dallo stesso ordine iniziale delle
carte. Ogni partita produce:

- esito e numero di boss sconfitti;
- numero di decisioni e fasi di difesa;
- reward cumulativa;
- yield e azioni invalide;
- tempo totale;
- latenza media e 95° percentile delle decisioni;
- stato di completamento o superamento del limite di decisioni.

La pipeline misura i metodi in sequenza sulla stessa macchina. Per confronti
temporali affidabili è opportuno chiudere applicazioni con carico variabile e
annotare l'hardware usato.

## Analisi statistiche

- percentuale di vittorie: intervallo di confidenza di Wilson;
- metriche continue: intervallo bootstrap della media;
- differenza nelle vittorie tra coppie: test esatto di McNemar;
- differenza nei boss sconfitti: test di Wilcoxon appaiato;
- confronti multipli: correzione di Holm;
- dimensione dell'effetto: Cohen *dz* e probabilità di superiorità.

I p-value non vengono interpretati da soli: il report presenta anche intervalli
di confidenza, dimensione dell'effetto e costo computazionale.

## Output

Ogni esecuzione crea una run sotto:

```text
artifacts/runs/<data>/experimental-report-<id>/
├── config.yaml
├── experimental_report_config.yaml
├── datasets/
│   └── games.csv
├── metrics/
│   └── metrics.jsonl
└── analysis/
    ├── experimental_report.md
    ├── summary.csv
    ├── pairwise_tests.csv
    ├── statistics.json
    ├── tables.md
    ├── tables.tex
    ├── comprehensive_dashboard.png
    ├── win_rate.png
    ├── bosses_defeated.png
    ├── execution_time.png
    └── quality_cost_tradeoff.png
```

La copia della configurazione conserva i parametri originali, mentre
`experimental_report_config.yaml` conserva la configurazione effettiva dopo le
eventuali opzioni CLI.

## Rigenerare il report

Grafici e tabelle possono essere rigenerati dai dati raw senza ripetere le
partite:

```bash
python -m scripts.experimental_report.analysis \
  artifacts/runs/<data>/<run_id>
```

Per produrre soltanto `datasets/games.csv`:

```bash
python -m scripts.experimental_report.runner --agents random heuristic
```
