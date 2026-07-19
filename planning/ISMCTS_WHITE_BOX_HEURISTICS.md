# Distillazione di ISMCTS in euristiche white-box

## Scopo

Questo documento descrive un possibile sviluppo futuro: osservare le scelte di
un agente **ISMCTS** e trasformarle in una strategia compatta, interpretabile e
utilizzabile anche da un giocatore umano.

Le alternative di implementazione sono confrontate anche nella pagina
interattiva [`regicide-ai-routes.html`](../docs/regicide-ai-routes.html).

L'output desiderato non è un altro agente opaco che imita ISMCTS, ma un insieme
ordinato di istruzioni del tipo:

- «nei primi boss conserva i fiori, salvo che permettano una sconfitta esatta»;
- «se l'attacco residuo del nemico è alto, preferisci le picche non immuni»;
- «conserva carte di valore basso quando aumentano la probabilità di una combo
  utile nei turni successivi».

Queste frasi sono **ipotesi illustrative**, non regole già dimostrate dai dati.
In particolare, una preferenza generale per o contro un seme può essere
facilmente smentita da immunità, composizione della mano e opportunità di
chiudere il boss.

Il nome corretto dell'algoritmo presente nel progetto è **ISMCTS**
(*Information Set Monte Carlo Tree Search*), non ISMTCS.

## Obiettivo e non-obiettivi

L'obiettivo è produrre un agente e, contemporaneamente, una piccola guida
strategica con queste proprietà:

- **white-box**: ogni decisione è riconducibile a una regola leggibile;
- **eseguibile da una persona**: le condizioni usano soltanto concetti
  osservabili al tavolo e calcoli semplici;
- **fedele all'esperto**: nelle situazioni coperte sceglie azioni simili a
  ISMCTS;
- **competitivo**: conserva una parte significativa della forza di ISMCTS pur
  essendo molto più economico;
- **onesto sull'incertezza**: distingue regole robuste da tendenze deboli o
  basate su pochi casi.

## Quanto è corretta l'analogia con AlphaZero

Definire l'idea «una specie di AlphaZero più manuale» è una buona intuizione,
ma richiede una precisazione.

L'analogia è valida nel ciclo generale:

1. una ricerca costosa produce una politica migliore;
2. un modello più economico apprende dalle decisioni della ricerca;
3. il modello viene valutato giocando;
4. i suoi errori generano nuovi stati da sottoporre all'esperto.

Questo ricorda la *policy improvement* di AlphaZero e, ancora più direttamente,
la **expert iteration**: una ricerca fa da esperto e un modello fa da
apprendista.

La proposta non è però AlphaZero in senso tecnico. AlphaZero usa normalmente
una rete neurale di policy e valore, addestrata da self-play, e riutilizza la
rete dentro MCTS. Qui:

- l'esperto iniziale è l'ISMCTS già disponibile;
- l'apprendista è intenzionalmente interpretabile, per esempio un albero
  decisionale corto o una lista ordinata di regole;
- il prodotto finale è una politica simbolica leggibile, non una rete;
- Regicide presenta informazione nascosta, quindi l'esperto opera su
  *information set* e determinizazioni.

La definizione più precisa è quindi:

> **distillazione interpretabile di una policy ISMCTS con ciclo di expert
> iteration**.

Si può usare «AlphaZero manuale» come metafora divulgativa, purché non venga
presentato come una replica dell'algoritmo AlphaZero.

## Perché il problema è difficile

### Spazio combinatorio e sparsità

Le combinazioni non sono letteralmente infinite, ma sono abbastanza numerose da
rendere quasi unico ogni stato grezzo: mano, boss, salute residua, protezione,
carte già viste, taverna, scarti, fase e azioni disponibili interagiscono tra
loro. Una tabella `stato esatto → azione` avrebbe quindi copertura minima e non
produrrebbe conoscenza riutilizzabile.

La soluzione non è raccogliere soltanto più partite. Occorre sostituire gli
stati grezzi con concetti strategici condivisi da molti stati:

- fase di attacco o difesa;
- inizio, metà o fine partita;
- tier e seme del boss;
- pressione del nemico: attacco e salute residui;
- quantità e qualità delle risorse;
- composizione della mano per valori, semi e combo disponibili;
- possibilità di exact kill, exact defense o attivazione utile di un potere.

### Frequenza non significa preferenza

Se ISMCTS gioca raramente picche, non segue automaticamente che «le picche sono
da evitare». Potrebbero essere apparse raramente in mano, essere state immuni o
non essere state legali nelle situazioni osservate.

Ogni analisi deve quindi confrontare una scelta con le **alternative realmente
disponibili** in quello stato. Le statistiche principali devono essere
normalizzate per opportunità, non soltanto per numero totale di azioni.

### Una scelta non basta a spiegare il motivo

La sola azione finale perde gran parte dell'informazione della ricerca. Due
azioni possono essere quasi equivalenti, oppure ISMCTS può preferirne una con
un margine enorme. Senza visite, valori e disponibilità delle alternative, il
modello tratterebbe allo stesso modo decisioni sicure e decisioni arbitrarie.

### Le regole possono entrare in conflitto

«Conserva i fiori» e «realizza una sconfitta esatta» possono suggerire azioni
opposte. L'output non dovrebbe essere una collezione non ordinata di slogan,
ma una **lista con priorità ed eccezioni**.

## Unità di analisi: la decisione ISMCTS

Il dataset futuro dovrebbe avere una riga principale per decisione e una
tabella collegata con una riga per ogni alternativa legale.

### Contesto osservabile

Salvare soltanto informazioni accessibili al giocatore:

- identificatori di run, partita, seed, turno e decisione;
- fase di attacco o difesa;
- boss corrente, tier, seme, salute e attacco residui;
- immunità corrente ed eventuale annullamento del giullare;
- protezione accumulata e difesa richiesta;
- dimensione di mano, taverna e scarti;
- carte in mano e carte pubblicamente note;
- indicatori derivati come combo disponibili, exact kill ed exact defense;
- insieme delle azioni legali.

Non devono diventare feature della policy:

- ordine reale delle carte nascoste;
- singole determinizazioni campionate dalla ricerca;
- qualunque informazione disponibile al simulatore ma non al giocatore.

### Diagnostica dell'esperto

Per ogni alternativa legale registrare almeno:

- identificatore e descrizione semantica dell'azione;
- numero di visite;
- conteggio di disponibilità usato da ISMCTS;
- ricompensa media stimata;
- quota delle visite alla radice;
- azione infine selezionata.

Registrare inoltre budget di ricerca, costante di esplorazione, seed casuali e
versione dell'agente. Poiché l'implementazione corrente può riutilizzare parte
dell'albero tra turni, i valori destinati all'analisi devono rappresentare lo
**snapshot della singola decisione**: occorre distinguere visite pregresse e
visite aggiunte dalla ricerca corrente, oppure eseguire una modalità di
raccolta con radice nuova. In caso contrario la policy target può essere
contaminata dalla storia dell'albero.

### Esito

Collegare la decisione sia a risultati locali sia all'esito della partita:

- danno, protezione, pesca e carte consumate;
- exact kill, overkill, exact defense e over-defense;
- boss sconfitto o morte nel seguito immediato;
- progresso totale e vittoria finale.

Questi campi servono per validare le regole, non per sostituire il giudizio
della ricerca con correlazioni retrospettive fragili.

## Vocabolario semantico

Prima di apprendere regole occorre definire un piccolo linguaggio condiviso tra
dati, modello e guida per il giocatore.

### Stato

Le feature dovrebbero privilegiare quantità comprensibili:

| Gruppo | Esempi |
| --- | --- |
| Progresso | boss 1–4 / 5–8 / 9–12, tier J/Q/K |
| Pericolo | attacco residuo basso/medio/alto, difesa richiesta |
| Risorse | carte in mano, carte basse/alte, numero di semi distinti |
| Opportunità | exact kill disponibile, exact defense disponibile, combo legali |
| Poteri | picche efficaci, fiori efficaci, pesca possibile, recupero possibile |
| Rischio | numero di azioni legali, margine di difesa, mano dopo la mossa |

Le soglie non devono essere scelte solo perché producono un albero più accurato:
vanno arrotondate a valori che una persona può riconoscere rapidamente.

### Azione

Gli ID tecnici devono essere tradotti in categorie:

- yield, giullare, carta singola o combo;
- attacco o difesa;
- valore totale e numero di carte;
- semi ed effetti realmente attivi, tenendo conto dell'immunità;
- exact, eccesso o difetto rispetto all'obiettivo;
- risorsa strategica consumata o conservata.

Due azioni diverse possono essere equivalenti per la guida umana. Se la regola
dice «usa una picche bassa», non è sempre necessario predire l'esatto ID della
carta.

## Target di apprendimento

La distribuzione di visite alla radice è un target più informativo della sola
azione vincente. Mostra sia la preferenza sia l'incertezza dell'esperto.

Sono utili tre target complementari:

1. **azione o categoria preferita**, per costruire la policy finale;
2. **preferenze a coppie** fra alternative legali, per imparare frasi come
   «a parità di exact kill, conserva il valore più alto»;
3. **confidenza**, ricavata da quota della prima azione, gap di valore ed
   entropia delle visite.

Le decisioni con poco budget, poche visite o margine quasi nullo possono essere
pesate meno. Non vanno trasformate in regole categoriche.

Il conteggio di disponibilità è essenziale in ISMCTS: una visita bassa non è
comparabile tra azioni che sono state disponibili in un numero molto diverso
di determinizazioni.

## Estrazione delle regole

È preferibile procedere dal modello interpretabile più semplice.

### 1. Separare problemi diversi

Costruire almeno due policy:

- attacco;
- difesa.

Successivamente si può separare anche per tier del boss, ma soltanto se il
supporto statistico è sufficiente. Un unico modello globale rischia di
mescolare obiettivi incompatibili.

### 2. Apprendere un modello white-box

Possibili candidati, in ordine pratico:

- albero decisionale poco profondo;
- lista ordinata di regole;
- *decision set* con priorità esplicite;
- modello additivo interpretabile per individuare tendenze, poi convertito in
  regole.

Una foresta o una rete può essere usata come diagnostica per stimare quanta
informazione si perde, ma non dovrebbe essere il prodotto finale: spiegare a
posteriori un modello opaco non equivale ad avere una policy white-box.

### 3. Comprimere e riscrivere

Le regole apprese devono essere:

- unite quando differiscono solo per soglie vicine;
- eliminate se hanno supporto insufficiente;
- ordinate per priorità;
- corredate da eccezioni rilevanti;
- riscritte con il vocabolario del gioco;
- limitate a pochi confronti numerici.

Formato consigliato:

> **Quando** [condizioni osservabili], **preferisci** [categoria di azione],
> **perché** [effetto strategico stimato].
>
> **Eccezione:** [condizione più importante].
>
> Supporto: [numero di decisioni]; fedeltà locale: [stima con intervallo].

### 4. Risolvere i conflitti

Una guida eseguibile richiede una gerarchia, per esempio:

1. evita la sconfitta immediata;
2. sfrutta una chiusura esatta affidabile;
3. attiva il potere necessario contro la minaccia corrente;
4. conserva risorse per i turni successivi;
5. usa la regola di fallback.

Questa gerarchia è soltanto una struttura iniziale: priorità e contenuto devono
essere appresi e poi verificati.

## Come affrontare la sparsità

La pipeline dovrebbe combinare più accorgimenti:

- **astrazione semantica** invece di identificatori di stato;
- **normalizzazione per opportunità**, confrontando solo azioni legali;
- **supporto minimo** prima di pubblicare una regola;
- **pooling gerarchico**, partendo da regole globali e specializzandole solo
  quando i dati giustificano l'eccezione;
- **binning stabile** di salute, attacco e risorse;
- **regolarizzazione verso modelli corti**;
- **intervalli di incertezza** tramite bootstrap a livello di partita;
- **raccolta mirata** di nuovi stati dove il modello è incerto o non concorda
  con ISMCTS.

Il train/test split deve avvenire per partita o seed, mai distribuendo decisioni
della stessa traiettoria tra training e test. In caso contrario stati quasi
duplicati produrrebbero una stima di fedeltà troppo ottimistica.

## Ciclo di expert iteration interpretabile

Il processo futuro può essere iterativo:

1. eseguire ISMCTS con budget alto su un insieme diversificato di partite;
2. creare il dataset semantico delle decisioni;
3. apprendere una prima lista di regole;
4. giocare con la policy white-box;
5. individuare sconfitte, stati non coperti e forti disaccordi con ISMCTS;
6. interrogare di nuovo ISMCTS soprattutto su quegli stati;
7. aggiornare, semplificare e rivalidare le regole.

L'apprendista non deve necessariamente essere inserito nei rollout di ISMCTS.
Farlo potrebbe migliorare entrambi, ma introdurrebbe un secondo esperimento e
renderebbe più difficile capire se il progresso deriva dalla distillazione o
dal cambiamento dell'esperto. La prima versione dovrebbe mantenere l'esperto
fisso.

## Valutazione

Nessuna singola metrica è sufficiente.

### Fedeltà all'esperto

- accordo top-1 sulla categoria e sull'azione;
- quota di visite ISMCTS assegnata all'azione scelta dalla regola;
- regret stimato rispetto alla migliore alternativa;
- fedeltà per attacco, difesa, tier, seme e livello di confidenza.

### Forza di gioco

- win rate e boss sconfitti;
- confronto a seed appaiati con ISMCTS e con l'HeuristicAgent attuale;
- robustezza su seed mai usati per estrarre le regole;
- latenza per decisione.

Una policy può avere accordo top-1 moderato ma forza elevata se diverge soltanto
fra mosse equivalenti. Per questo il regret e il risultato di gioco sono più
informativi della sola accuracy.

### Interpretabilità operativa

- numero totale di regole;
- profondità massima e numero di condizioni per regola;
- percentuale di decisioni coperta;
- numero di concetti e soglie da ricordare;
- frequenza di conflitti o fallback;
- tempo richiesto a una persona per applicare la guida;
- accordo tra persone nell'interpretazione della stessa regola.

Un piccolo test con giocatori è importante: una regola formalmente leggibile
può comunque essere troppo lenta da applicare durante la partita.

## Controlli e ablation

Per capire quali elementi sono realmente utili:

- scelta finale contro distribuzione completa delle visite;
- feature grezze contro feature semantiche;
- target azione esatta contro categoria di azione;
- modello unico contro policy separate attacco/difesa;
- regole senza confidenza contro regole pesate per confidenza;
- dataset passivo contro raccolta attiva sugli stati di disaccordo;
- budget ISMCTS basso, medio e alto come qualità dell'insegnante.

## Rischi principali

| Rischio | Conseguenza | Mitigazione |
| --- | --- | --- |
| Imitare errori o rumore di ISMCTS | Regole semplici ma sbagliate | Esperto ad alto budget, confidenza e valutazione autonoma |
| Confondere disponibilità e preferenza | Regole distorte sui semi o sulle combo | Salvare tutte le alternative legali e l'availability count |
| Usare informazioni nascoste | Guida impossibile per un umano | Allowlist esplicita delle sole feature osservabili |
| Regole troppo specifiche | Copertura bassa | Astrazione, supporto minimo e pooling gerarchico |
| Regole troppo generiche | Perdita di forza | Eccezioni con supporto e raccolta mirata |
| Leakage tra decisioni della stessa partita | Metriche di fedeltà gonfiate | Split per seed o partita |
| Correlazione scambiata per causalità | Consigli strategici fuorvianti | Confronto tra alternative e verifica controfattuale con ricerca |
| Troppe regole | Agente leggibile ma guida inutilizzabile | Budget esplicito di complessità e test con giocatori |

## Fasi di una possibile implementazione futura

| Fase | Risultato atteso |
| --- | --- |
| WBOX-01 — Contratto | Definizione di osservabilità, vocabolario e formato delle regole |
| WBOX-02 — Tracing | Dataset per decisione e alternativa con snapshot ISMCTS riproducibile |
| WBOX-03 — Baseline | Misura della fedeltà ottenibile con modelli semplici |
| WBOX-04 — Distillazione | Policy separate attacco/difesa e prima lista ordinata di regole |
| WBOX-05 — Validazione | Benchmark a seed appaiati, incertezza e analisi degli errori |
| WBOX-06 — Iterazione attiva | Nuovi stati raccolti su disaccordi e casi non coperti |
| WBOX-07 — Guida umana | Riscrittura finale e test di applicabilità con giocatori |

## Primo esperimento consigliato

Per ridurre il rischio, il primo studio dovrebbe essere piccolo:

1. usare un solo ISMCTS fisso e ad alto budget;
2. raccogliere soltanto feature pubbliche, azioni legali e statistiche della
   radice;
3. limitarsi alle decisioni di attacco;
4. predire categorie ampie, non l'ID esatto;
5. apprendere un albero molto corto e convertirlo in lista di regole;
6. valutarlo su partite e seed separati;
7. verificare manualmente le cinque regole con maggiore supporto.

Questo esperimento risponde presto alla domanda decisiva: esiste una parte
sostanziale della strategia ISMCTS che può essere compressa in istruzioni
semplici senza perdere quasi tutta la forza di gioco?

## Criteri di successo

La direzione è promettente se il prototipo:

- supera chiaramente l'HeuristicAgent attuale su seed non osservati;
- conserva una quota utile delle prestazioni di ISMCTS;
- copre la maggioranza delle decisioni con un numero contenuto di regole;
- non usa alcuna informazione nascosta;
- mostra supporto e incertezza per ogni regola pubblicata;
- produce istruzioni che un giocatore riesce ad applicare in modo coerente.

Il risultato negativo sarebbe comunque informativo: indicherebbe quali parti
della strategia richiedono pianificazione profonda e non possono essere ridotte
onestamente a euristiche locali.
