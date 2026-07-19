# Piano di azione per i punti 2–5 del TO-DO

## Scopo

Questo documento traduce i punti 2–5 di [`TO-DO.md`](TO-DO.md) in un piano
eseguibile basato sullo stato reale del repository al 18 luglio 2026.

Il punto 1, «Refactoring e pulizia del codice», è intenzionalmente escluso.
Le modifiche architetturali indicate qui sono limitate a quanto necessario per
documentazione, logging, raccolta sperimentale e pubblicazione.

## Sintesi esecutiva

Il progetto non parte da zero: esistono già `pdoc`, un `ml_logger` proprietario,
un catalogo delle run, la registrazione delle partite e una pipeline sperimentale
con test statistici e plot. Il lavoro raccomandato è quindi consolidare queste
parti, collegarle tra loro e rimuovere le discrepanze tra interfacce dichiarate e
comportamento effettivo.

| Punto | Stato attuale | Decisione raccomandata | Impatto principale |
| --- | --- | --- | --- |
| 2. Documentazione | Copertura disomogenea; generatore già presente ma parziale | Google Style, policy verificabile e API docs separate dai documenti manuali | Migliore comprensibilità e minore rischio di documentazione obsoleta |
| 3. Logging | Persistenza funzionante; dashboard separata e mai avviata dagli script | Un solo backend, con dashboard opzionale come vista del `RunContext` | CPU/GPU/RAM visibili e metriche live coerenti con quelle salvate |
| 4. Report | Buona base aggregata e dati sufficienti per pochi plot semplici | Mantenere solo le viste già supportate e migliorarne il design | Risultati più chiari senza nuova strumentazione o nuovi dataset |
| 5. README | Un file già ricco ma troppo esteso e con alcuni riferimenti non validi | Tre documenti primari: `README.md`, `docs/USAGE.md`, `docs/RESULTS.md` | Onboarding più rapido e risultati pubblicabili/riproducibili |

## Evidenze usate per l'analisi

- [`scripts/generate_docs.py`](../scripts/generate_docs.py) usa già Google
  Style con `pdoc`, ma documenta soltanto `game`, `agents` e `ml_logger` e
  scrive direttamente in `docs/`.
- [`ml_logger/runtime.py`](../ml_logger/runtime.py) configura il logging
  standard e la persistenza delle run; la dashboard vive invece in
  [`ml_logger/core/logger.py`](../ml_logger/core/logger.py).
- `DashboardLogger` è esportato, ma non è istanziato da alcun workflow del
  progetto. CPU, GPU e RAM sono aggiornate soltanto nel suo ciclo `Live`.
- In [`logger_config.yaml`](../logger_config.yaml) `saving.telemetry` è
  `false`, quindi la telemetria non viene neppure storicizzata.
- La pipeline in
  [`scripts/experimental_report/`](../scripts/experimental_report/) salva
  metriche aggregate per partita, applica un protocollo a seed appaiati,
  supporta resume e produce intervalli di confidenza e test appaiati.
- Il runner sperimentale crea `RegicideEnv` senza `GameRecorder`; il dataset
  `games.csv` non contiene la sequenza delle azioni.
- L'ultima run disponibile,
  `experimental-report-20260718T160605-4dcbb706`, contiene 400 partite:
  100 per Random, Heuristic, PIMC e ISMCTS. I risultati aggregati principali
  sono 42% di vittorie e 10,01 boss medi per ISMCTS, 24% e 9,25 per PIMC,
  0% e 4,37 per Heuristic, 0% e 2,97 per Random.
- La stessa run è marcata `failed`: al momento di aggiungere PPO, il loader ha
  cercato `artifacts/promoted_models/ppo_regicide.zip.zip`. I plot esistenti
  descrivono quindi una run parziale, non una run finale completata.
- [`README.md`](../README.md) contiene già molte informazioni, ma conserva
  l'URL segnaposto `yourusername`; [`ui/README.md`](../ui/README.md) indica
  `python -m ui.app`, mentre l'entry point reale è `python -m ui`.

---

## 2. Documentazione del codice

### 2.1 Analisi dello stato attuale

Un controllo AST sui package applicativi produce questa baseline. Nel conteggio
«funzioni pubbliche» sono inclusi anche i metodi pubblici; la presenza di una
docstring non ne misura ancora la qualità.

| Package | Moduli documentati | Classi documentate | Funzioni pubbliche documentate |
| --- | ---: | ---: | ---: |
| `game` | 3/4 | 1/5 | 22/29 |
| `agents` | 4/9 | 8/8 | 7/12 |
| `solvers` | 17/31 | 11/13 | 32/54 |
| `ml_logger` | 6/21 | 11/13 | 29/76 |
| `scripts` | 14/16 | 1/1 | 19/24 |
| `ui` | 1/6 | 0/4 | 0/23 |
| **Totale** | **45/87** | **32/44** | **109/218** |

Gap rilevanti:

- le entità centrali `Suit`, `Card`, `Enemy` e `Game` non hanno una docstring
  di classe;
- diversi contratti pubblici non sono descritti, tra cui `RegicideEnv.reset`,
  alcuni `select_action`, i metodi di `RunContext` e il ciclo di vita della
  dashboard;
- molte docstring esistenti descrivono il “cosa”, ma non invarianti, effetti
  collaterali, eccezioni o ragioni delle scelte algoritmiche;
- `scripts/generate_docs.py` esclude `solvers`, `scripts` e `ui`;
- i file HTML generati e la documentazione manuale condividono `docs/`, con
  rischio di sovrascritture e pagine obsolete;
- non esiste un controllo automatico della copertura o del formato delle
  docstring.

### 2.2 Standard proposto

Adottare **Google Style**, perché è già il formato configurato in `pdoc`.
Cambiare standard ora aggiungerebbe lavoro senza un vantaggio concreto.

Una funzione è considerata non banale se soddisfa almeno una di queste
condizioni:

- contiene regole di dominio, branching significativo o mutazione dello stato;
- esegue I/O, persistenza, multiprocessing o gestione del ciclo di vita;
- implementa un algoritmo, una metrica, un protocollo statistico o una
  trasformazione non ovvia;
- espone un contratto usato da un altro modulo;
- ha precondizioni, effetti collaterali, fallback o failure mode che un lettore
  non può dedurre facilmente dalla firma.

Sono escluse dall'obbligo le semplici proprietà, i `dunder` ovvi e i wrapper di
una riga, salvo comportamento sorprendente.

Ogni docstring non banale deve contenere, quando applicabile:

- sintesi del ruolo e motivazione;
- `Args`, `Returns`, `Raises`;
- mutazioni ed effetti collaterali;
- invarianti di Regicide o assunzioni sul formato dei dati;
- `Note` per complessità, determinismo, thread/process safety;
- un esempio soltanto per API pubbliche il cui uso non è evidente.

### 2.3 Cambiamenti proposti

1. Creare una policy breve in `docs/DOCUMENTATION.md` con standard, esempi e
   Definition of Done.
2. Documentare prima le API stabili (`game`, `agents`, `solvers/env.py`), poi
   `ml_logger` e la pipeline sperimentale dopo i cambiamenti dei punti 3 e 4.
3. Estendere la generazione a tutti i package pubblicabili.
4. Scrivere l'output generato in `docs/api/`, lasciando `docs/` per i documenti
   curati manualmente.
5. Aggiungere un controllo AST o `pydocstyle` che fallisca per classi e API
   pubbliche prive di docstring; le funzioni private non banali restano parte
   della review manuale.
6. Aggiungere uno smoke test che costruisca la documentazione e controlli link,
   import falliti e pagine principali.

### 2.4 Impatti

**Codice.** Le modifiche sono prevalentemente additive, ma scrivere docstring
complete farà emergere firme ambigue e contratti impliciti. Questi problemi
vanno segnalati, non corretti dentro questo punto se appartengono al refactoring
generale escluso.

**Build.** La generazione completa importerà moduli con dipendenze opzionali
come PyTorch, Stable-Baselines3 e Pygame. Il build deve fallire con un messaggio
chiaro oppure usare una configurazione documentata per gli import opzionali.

**Repository.** Separare `docs/api/` evita collisioni con
`docs/experimental_report.md` e altre pagine manuali. Va deciso se versionare
l'HTML o pubblicarlo come artefatto; la raccomandazione è generarlo in CI e
versionare soltanto le sorgenti Markdown.

**Manutenzione.** Il controllo automatico aumenta leggermente il costo di ogni
nuova API, ma impedisce di tornare rapidamente alla copertura attuale.

### 2.5 Ticket

| ID | Attività | Output e Definition of Done | Dipendenze | Priorità | Impegno |
| --- | --- | --- | --- | --- | --- |
| DOC-01 | Definire policy Google Style e criterio “non banale” | `docs/DOCUMENTATION.md` approvato, con template ed esempi presi da gioco, logger e report | Nessuna | P0 | Piccolo |
| DOC-02 | Aggiungere audit automatico | Report riproducibile per package; test che segnala API pubbliche mancanti senza falsi positivi noti | DOC-01 | P0 | Medio |
| DOC-03 | Documentare dominio e agenti | `game`, `agents` e `solvers/env.py` coperti; invarianti di carte, azioni e osservazioni espliciti | DOC-01 | P0 | Grande |
| DOC-04 | Documentare logger e report | Ciclo di vita, persistenza, multiprocessing, schemi dati e failure mode descritti | LOG-04, REP-04 | P1 | Grande |
| DOC-05 | Isolare e completare il build `pdoc` | Tutti i package selezionati generano pagine in `docs/api/`; nessuna pagina manuale viene sovrascritta | DOC-03, DOC-04 | P1 | Medio |
| DOC-06 | Verificare la documentazione in automazione | Comando unico di build, smoke test verde e controllo link/import | DOC-05 | P1 | Medio |

---

## 3. Sistema di logging e dashboard

### 3.1 Come viene usato oggi

Il flusso prevalente è:

1. un entry point chiama `start_run(...)`;
2. viene creato un `RunContext` con manifest, directory della run e catalogo
   SQLite;
3. `configure_logging(...)` collega il logging Python a console Rich e
   `logs/run.log`;
4. lo script usa `get_logger(__name__)`, `context.log_metrics(...)`,
   `context.save_result(...)` e infine `context.complete(...)` o
   `context.fail(...)`.

Questo flusso salva correttamente log, metriche e risultati, ma **non avvia la
dashboard**.

`DashboardLogger` è una seconda facciata. Solo il suo metodo `start()` crea il
`rich.live.Live`, e solo il callback `_update_and_get_layout()` legge CPU, RAM e
GPU. Nessuno script istanzia o avvia questa classe. Inoltre:

- `context.log_metrics(...)` non aggiorna il dizionario in memoria usato dal
  pannello «Metrics»;
- `DashboardLogger.update_metrics(...)` aggiorna il pannello, ma non è usato
  dai solver;
- `saving.telemetry: false` disabilita `telemetry.jsonl`;
- il campionamento GPU invoca `nvidia-smi` e restituisce stringhe, poco adatte
  ad analisi successive;
- il valore RAM è di sistema, non la memoria del processo della run.

Quindi la causa della dashboard assente non è un problema di rendering: la
dashboard appartiene a un percorso di esecuzione che i comandi correnti non
percorrono.

### 3.2 Architettura proposta

Mantenere il logging standard come unica API applicativa e trasformare la
dashboard in una vista opzionale del medesimo `RunContext`.

```text
entry point
    |
    v
run_scope(...) ---> RunContext ---> manifest / log / metrics / results
    |                    |
    |                    +-------> stream di eventi metrici
    |
    +---- dashboard opzionale ---> Live Rich + sampler hardware
                                  |
                                  +--> telemetry.jsonl
```

API raccomandata:

```python
with run_scope("experimental-report", dashboard="auto") as context:
    logger.info("...")
    context.log_metrics(step, metrics)
```

`start_run` può restare disponibile per compatibilità. `run_scope` deve
garantire `complete`, `fail`, stop della dashboard e chiusura degli handler.
La dashboard non deve introdurre un secondo metodo di logging.

Configurazione proposta:

```yaml
dashboard:
  enabled: auto          # auto, true, false
  refresh_rate: 4
  max_log_lines: 50
  fallback: compact      # terminale non interattivo

telemetry:
  enabled: true
  sample_interval_sec: 2
  persist_interval_sec: 10
  include_process: true
  include_system: true
  gpu_provider: auto
```

`auto` abilita la vista solo su terminale interattivo e nel processo padre.
Test, CI, redirect su file e worker multiprocess devono usare il fallback
compatto.

### 3.3 Telemetria proposta

Salvare valori numerici strutturati:

- CPU di sistema e CPU del processo;
- RAM di sistema, RSS e picco RSS del processo;
- GPU utilization, memoria usata/totale, device e temperatura se disponibili;
- timestamp monotono/UTC e step corrente;
- stato «provider non disponibile» distinto dal valore zero.

Il sampling deve avvenire in background o essere memorizzato in cache. Eseguire
`nvidia-smi` quattro volte al secondo dentro il render può alterare proprio le
misure che si vogliono osservare.

Per workflow multiprocess:

- il processo padre è l'unico proprietario del display e dei file aggregati;
- i worker inviano eventi strutturati tramite coda o restituiscono metriche al
  parent;
- la dashboard mostra anche numero di worker attivi, partite completate e
  throughput;
- la telemetria distingue il processo padre dal consumo complessivo della
  macchina.

### 3.4 Impatti

**Entry point.** Gli script devono migrare gradualmente a `run_scope`. Non è
necessario cambiare le chiamate `logger.info(...)`.

**Prestazioni.** Il sampler introduce overhead, da misurare con dashboard on/off.
Il default `auto` e intervalli separati per render e persistenza lo limitano.

**Multiprocessing.** Rich Live deve rimanere nel parent; scritture concorrenti
dirette renderebbero output e metriche instabili.

**Compatibilità.** Manifest e `metrics.jsonl` restano validi. La telemetria
richiede uno schema versionato perché sostituisce le stringhe GPU con campi
numerici.

**Test.** Servono test di lifecycle, terminale non interattivo, assenza GPU,
eccezioni, handler duplicati e consistenza fra valori live e persistiti.

### 3.5 Ticket

| ID | Attività | Output e Definition of Done | Dipendenze | Priorità | Impegno |
| --- | --- | --- | --- | --- | --- |
| LOG-01 | Formalizzare il contratto run/log/dashboard | Nota architetturale con ownership, lifecycle e schema eventi; nessuna doppia API applicativa | Nessuna | P0 | Piccolo |
| LOG-02 | Introdurre `run_scope` compatibile | Context manager testato; completa o fallisce la run e chiude dashboard/handler anche su eccezione | LOG-01 | P0 | Medio |
| LOG-03 | Collegare metriche e log alla vista live | `context.log_metrics` aggiorna sia storage sia pannello, senza chiamate duplicate negli script | LOG-02 | P0 | Grande |
| LOG-04 | Rifattorizzare la dashboard come vista | Modalità `auto/true/false`, fallback non TTY e ownership esclusiva del parent | LOG-03 | P0 | Grande |
| LOG-05 | Strutturare il sampler hardware | CPU/RAM processo+sistema, GPU numerica opzionale, cache e schema versionato | LOG-04 | P1 | Medio |
| LOG-06 | Migrare gli entry point principali | `log_game`, training, benchmark e report mostrano metriche coerenti e non usano `DashboardLogger` direttamente | LOG-04 | P1 | Grande |
| LOG-07 | Testare overhead e robustezza | Test automatici verdi; benchmark on/off documentato; nessun handler o thread residuo | LOG-05, LOG-06 | P1 | Medio |

---

## 4. Report sperimentale e visualizzazioni

### 4.1 Stato attuale

La pipeline corrente è una buona base:

- configurazione centralizzata degli agenti;
- stessi seed per ogni agente;
- checkpoint atomico dopo ogni partita;
- resume e parallelismo;
- Wilson CI, bootstrap, McNemar, Wilcoxon, correzione di Holm ed effect size;
- tabelle CSV/Markdown/LaTeX;
- plot di win rate, boss sconfitti, tempo e trade-off qualità/costo.

`games.csv` contiene già outcome, boss sconfitti, turni, durata, latenza delle
decisioni, yield, errori e reward. Queste informazioni sono sufficienti per il
report corrente.

I plot esistenti raccontano la stessa storia aggregata da angolazioni in parte
ridondanti. Il prossimo intervento non deve aumentare il numero di grafici o
inventare analisi non sostenute dai dati: deve selezionare poche viste e
migliorarne gerarchia visiva, consistenza e leggibilità.

### 4.2 Decisione di scope

Per questa fase:

- usare soltanto colonne già presenti in `games.csv` e statistiche già
  calcolate dalla pipeline;
- non aggiungere logging per decisione, tassonomie di azioni o diagnostica
  interna degli agenti;
- non introdurre proiezioni, network, Sankey o altre visualizzazioni
  esplorative;
- mantenere tre messaggi: probabilità di vittoria, progresso medio/distribuito
  e rapporto fra qualità e costo;
- considerare il dashboard soltanto una composizione delle stesse viste, non un
  quarto livello di analisi.

L'ispezione futura delle scelte ISMCTS e la distillazione in regole eseguibili
da un giocatore sono descritte separatamente in
[`ISMCTS_WHITE_BOX_HEURISTICS.md`](ISMCTS_WHITE_BOX_HEURISTICS.md). Quello
sviluppo richiederà dati nuovi, ma non fa parte dei plot o
dell'implementazione corrente.

### 4.3 Correzioni necessarie prima del report finale

#### Preflight della run

Prima di avviare centinaia di partite:

- validare tutti i `class_path`;
- istanziare ogni agente una volta;
- normalizzare e verificare i checkpoint, evitando il caso `.zip.zip`;
- stimare spazio disco e durata;
- salvare elenco di agenti `ready`, `disabled` e `failed`;
- interrompere prima della prima partita oppure escludere esplicitamente gli
  agenti non pronti, secondo una policy configurata.

Il report deve essere generato soltanto da una run `completed`, oppure mostrare
chiaramente un watermark/stato «partial». Una run fallita non deve sembrare un
risultato finale.

### 4.4 Plot confermati

| Visualizzazione | Domanda a cui risponde | Dati già disponibili |
| --- | --- | --- |
| **Win rate** | Quanto spesso vince ogni agente e quanto è incerta la stima? | `victory`, Wilson CI |
| **Bosses defeated** | Quanto avanza l'agente anche quando non vince? | `bosses_defeated`, media e distribuzione |
| **Quality versus cost** | Quanto progresso si ottiene rispetto al tempo richiesto? | media boss, durata, CI e win rate |

Il plot separato del tempo è opzionale nel report pubblico perché il trade-off
qualità/costo ne contiene già il messaggio principale. Può restare come
artefatto diagnostico della run.

### 4.5 Direzione di design

Il redesign deve applicare un unico sistema visivo:

- stesso colore per lo stesso agente in ogni figura;
- palette distinguibile anche con deficit della visione dei colori;
- ordinamento coerente, preferibilmente per efficacia;
- etichette dirette e nomi leggibili al posto di legende lontane;
- intervalli di confidenza visibili ma meno dominanti della stima centrale;
- assi con unità esplicite e range naturali (`0–100%` e `0–12` boss);
- scala logaritmica dichiarata quando necessaria per i tempi;
- titoli che esprimono il messaggio e sottotitoli che spiegano segni e
  intervalli;
- griglie leggere, alto contrasto del testo e spazio bianco sufficiente;
- annotazioni selettive, evitando di ripetere tutti i valori già presenti
  nelle tabelle;
- dimensioni adatte sia a `docs/RESULTS.md` sia alla lettura del singolo file.

Il dashboard, se mantenuto, deve riusare esattamente colori, ordinamenti e
tipografia dei plot individuali. Non deve duplicare il plot del costo e il
trade-off se lo spazio rende entrambi poco leggibili.

Gli artefatti finali rimangono separati dagli output delle run:

```text
artifacts/runs/<run_id>/analysis/   # plot rigenerabili della singola run
docs/assets/results/                # soli plot selezionati per la pubblicazione
docs/RESULTS.md                     # report pubblico
```

### 4.6 Protocollo sperimentale

Separare due misure:

1. **Qualità**: partite parallelizzate, seed appaiati, budget di ricerca
   comparabili, telemetria e checkpoint.
2. **Costo**: `jobs=1`, warm-up, macchina quanto più possibile inattiva,
   versione hardware/software salvata nel manifest.

Nell'ultima run `parallel_jobs=5`; i tempi di PIMC e ISMCTS sono quindi utili
come indicazione operativa, non come benchmark rigoroso.

Il report finale deve includere:

- protocollo e configurazione effettiva;
- commit, dirty state, hardware e versioni;
- agenti inclusi, esclusi e motivazione;
- sample size e seed;
- risultati con CI, effect size e test;
- limiti, run status e link al dataset;
- spiegazione sintetica dei tre plot selezionati.

### 4.7 Impatti

**Dati.** Non cambia lo schema di `games.csv` e non viene introdotto un dataset
decisionale.

**Prestazioni.** Nessun overhead nelle partite: il lavoro riguarda la
generazione successiva dei grafici.

**API agenti.** Nessuna modifica richiesta.

**Manutenzione.** Una palette e funzioni di stile condivise evitano divergenze
fra figure. I test devono concentrarsi su output deterministico, file generati
e presenza delle etichette essenziali.

**Validità.** Il redesign non deve suggerire precisione aggiuntiva. Intervalli,
sample size, stato della run e limiti del timing restano visibili.

### 4.8 Ticket

| ID | Attività | Output e Definition of Done | Dipendenze | Priorità | Impegno |
| --- | --- | --- | --- | --- | --- |
| REP-01 | Aggiungere preflight agenti e checkpoint | Tutti gli errori di classe/modello emergono prima delle simulazioni; test sul suffisso PPO | Nessuna | P0 | Medio |
| REP-02 | Consolidare il design dei plot | Palette, tipografia, assi, annotazioni e layout condivisi; nessuna nuova metrica | Nessuna | P0 | Medio |
| REP-03 | Selezionare e verificare i tre plot | Win rate, boss e qualità/costo deterministici e leggibili a dimensione report | REP-02 | P1 | Medio |
| REP-04 | Rendere il report status-aware | Nessun report “finale” da run failed; agenti esclusi e failure chiaramente visibili | REP-01 | P0 | Medio |
| REP-05 | Eseguire e pubblicare l'esperimento finale | Run `completed`, protocollo qualità+costo e soli plot selezionati in `docs/assets/results/` | REP-03, REP-04 | P1 | Grande |

---

## 5. Tre documenti primari del progetto

### 5.1 Analisi dello stato attuale

Il README corrente copre già installazione, comandi, logging, artifact layout,
documentazione e struttura. Il problema principale non è la quantità, ma la
mescolanza di tre pubblici diversi:

- visitatore che deve capire il progetto in pochi minuti;
- utente che vuole eseguire GUI, agenti, training e analisi;
- lettore interessato ai risultati scientifici.

Sono inoltre presenti segnali di drift:

- URL di clone segnaposto;
- comando UI incoerente tra README principale e `ui/README.md`;
- `TESTING.md` usa ancora percorsi `outputs/` per BC data, mentre la
  documentazione principale dichiara `artifacts/` come layout canonico;
- il README presenta il report come completo, ma l'ultima run è fallita prima
  di PPO;
- le metriche sono descritte manualmente e possono divergere dal report
  generato.

### 5.2 Struttura proposta

#### 1. `README.md` — progetto, installazione e orientamento

Obiettivo: permettere a un nuovo lettore di capire valore, maturità e accesso al
progetto in circa due minuti.

Contenuti:

- descrizione e obiettivo;
- feature realmente verificate;
- screenshot o singolo plot rappresentativo;
- requisiti e installazione minima;
- quickstart GUI e agente;
- struttura sintetica delle cartelle;
- dipendenze per gruppi, con `requirements.txt` come fonte autorevole;
- link a Usage, Results, API docs, test, licenza e regole.

Non deve contenere il manuale completo di tutti i comandi.

#### 2. `docs/USAGE.md` — esecuzione, funzionalità e agenti

Contenuti:

- matrice degli entry point;
- esempi di comandi verificati;
- GUI, simulazione, training, evaluation, report e resume;
- descrizione e requisiti di Random, Heuristic, PIMC, ISMCTS, PPO e AlphaZero;
- configurazione, seed e checkpoint;
- logging, dashboard e struttura degli artifact;
- troubleshooting, incluso modello assente/GPU assente/non-TTY;
- sviluppi futuri e limiti funzionali;
- collegamento a `TESTING.md` per i test di sviluppo.

#### 3. `docs/RESULTS.md` — risultati, metriche e plot

Contenuti:

- domanda sperimentale;
- protocollo, agenti e budget;
- identificativo della run `completed`, commit e hardware;
- tabelle principali;
- plot finali incorporati da `docs/assets/results/`;
- interpretazione, test statistici ed effect size;
- limiti e riproducibilità;
- comando per rigenerare il report.

I blocchi numerici devono essere generati dalla pipeline, non copiati a mano.
Uno script di pubblicazione deve rifiutare una run non completata.

### 5.3 Impatti

**Navigazione.** Il README diventa più corto e stabile; la profondità si sposta
su documenti dedicati con link reciproci.

**Automazione.** `RESULTS.md` dipende dal punto 4 e deve essere aggiornato da un
comando riproducibile. In questo modo run e documento pubblico non divergono.

**Drift.** I comandi mostrati devono essere smoke-testati. I documenti secondari
come `ui/README.md` vanno allineati o trasformati in brevi rimandi, senza creare
una quarta fonte autorevole.

**Lingua.** Per una repository pubblica si raccomanda inglese nei tre documenti
primari e italiano nei documenti di pianificazione. Se si desidera una release
bilingue, va aggiunta in seguito senza duplicare manualmente tabelle e metriche.

### 5.4 Ticket

| ID | Attività | Output e Definition of Done | Dipendenze | Priorità | Impegno |
| --- | --- | --- | --- | --- | --- |
| PUB-01 | Definire indice e fonti autorevoli | Outline dei tre file; ogni informazione dinamica ha una sola fonte | Nessuna | P0 | Piccolo |
| PUB-02 | Riscrivere il README landing | Nessun placeholder, quickstart verificato, link validi, lunghezza focalizzata | DOC-05, PUB-01 | P1 | Medio |
| PUB-03 | Creare `docs/USAGE.md` | Tutti gli entry point e agenti documentati; comandi smoke-testati; troubleshooting dashboard/modelli | LOG-06, PUB-01 | P1 | Grande |
| PUB-04 | Generare `docs/RESULTS.md` | Solo da run completata; tabelle/plot con provenance e limiti | REP-05, PUB-01 | P1 | Grande |
| PUB-05 | Allineare documenti secondari | `ui/README.md`, `TESTING.md` e protocollo sperimentale non contraddicono i tre file primari | PUB-02, PUB-03, PUB-04 | P1 | Medio |
| PUB-06 | Verificare release docs | Link check, smoke command e build API verdi su clone pulito | PUB-02, PUB-03, PUB-04, DOC-06 | P1 | Medio |

---

## Sequenza raccomandata

1. **Contratti e preflight:** DOC-01, LOG-01, REP-01, PUB-01.
2. **Fondazione osservabilità:** LOG-02–LOG-05.
3. **Design del report:** REP-02–REP-04, senza nuova raccolta decisionale.
4. **Migrazione entry point:** LOG-06.
5. **Documentazione del codice:** DOC-02–DOC-05, documentando per ultimi i
   moduli appena modificati.
6. **Run e pubblicazione:** REP-05, PUB-02–PUB-05.
7. **Gate finale:** LOG-07, DOC-06, PUB-06.

Questa sequenza evita di documentare due volte le API del logger e del report e
mantiene le visualizzazioni entro le informazioni già raccolte.

## Rischi trasversali e mitigazioni

| Rischio | Conseguenza | Mitigazione |
| --- | --- | --- |
| Il refactoring del punto 1 avviene in parallelo | Docstring e link diventano obsoleti | Documentare subito dominio stabile; logger/report soltanto dopo le rispettive modifiche |
| Dashboard e telemetry alterano i benchmark | Tempi non confrontabili | Sampler a bassa frequenza, misura overhead, protocollo costo con dashboard off |
| Scritture multiprocess concorrenti | Corruzione o duplicati | Parte atomica per agente+seed e consolidamento nel parent |
| Troppi plot aggregati e ridondanti | Il messaggio principale si disperde | Pubblicare tre viste coerenti e spostare i dettagli nelle tabelle |
| Modelli mancanti scoperti tardi | Ore di simulazione sprecate | Preflight obbligatorio prima della prima partita |
| Risultati copiati manualmente | README e artifact divergono | Generazione di `RESULTS.md` solo da run completata |

## Definition of Done complessiva

Il piano è completato quando:

- tutte le classi e funzioni non banali rispettano la policy scelta;
- la documentazione API si genera in una directory isolata senza errori;
- i principali entry point mostrano la dashboard in modalità `auto` su TTY e
  usano un fallback pulito altrove;
- CPU, GPU e RAM sono visibili e, quando abilitate, persistite come valori
  strutturati;
- una run sperimentale completata produce `games.csv` validato e resume sicuro;
- il report include soltanto i plot selezionati di win rate, boss sconfitti e
  qualità/costo, con design coerente e intervalli espliciti;
- tempi e qualità sono misurati con protocolli appropriati;
- `README.md`, `docs/USAGE.md` e `docs/RESULTS.md` hanno ruoli distinti, link
  validi e comandi verificati;
- suite di test, build documentazione e smoke test di pubblicazione sono verdi.

## Decisioni raccomandate da approvare prima dell'esecuzione

1. Google Style come standard unico.
2. Dashboard `auto` su TTY, non sempre attiva.
3. Nessun nuovo dataset decisionale nella fase corrente.
4. Tre plot semplici basati sui dati esistenti, con priorità al design.
5. Tre documenti primari in inglese, planning interno in italiano.
