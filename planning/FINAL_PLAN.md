# Obiettivo

Il progetto è da considerarsi **completo**: è stato sviluppato un agente basato su **ISMCTS** in grado di vincere una partita in solitario.

L'obiettivo ora non è più aggiungere nuove funzionalità, ma **preparare il progetto per la pubblicazione**, rendendolo pulito, ben documentato e facilmente utilizzabile da utenti esterni.

## 1. Refactoring e pulizia del codice

Dal punto di vista dell'architettura del gioco e degli agenti non sono previste nuove implementazioni, se non una revisione generale del codice.

Attività previste:

* Rimuovere file, script e codice non più utilizzati.
* Riorganizzare la struttura delle cartelle e dei file in modo più professionale e intuitivo.
* Analizzare ogni classe e ogni script singolarmente.
* Migliorare leggibilità, naming e organizzazione del codice.
* Eliminare duplicazioni creando funzioni o utility condivise quando opportuno.
* Standardizzare lo stile del codice e le convenzioni utilizzate in tutto il progetto.

---

## 2. Documentazione del codice

Creare una documentazione completa e facilmente generabile.

Obiettivi:

* Documentare tutte le classi e le funzioni con docstring complete e coerenti.
* Utilizzare uno standard di documentazione (ad esempio Google Style o NumPy Style).
* Predisporre la generazione automatica della documentazione a partire dalle docstring.

---

## 3. Sistema di logging

adatta ogni script per comunicare tramite il logger (ml_logger, remove others if present)
you can adapt the logger to fit this repo
Never use prints, always use the logger and save the results

Inoltre, desidero progettare separatamente un sistema che permetta di:

* salvare ogni run di gioco;
* storicizzare i risultati;
* poter analizzare una partita anche in un secondo momento.

è già presente qualcosa di simile ma è confuso e scritto male
crea un piano d'azione, discutiamo su come implementarlo e che feature inserire

---

## 4. Report sperimentale

Preparare il materiale necessario per descrivere e confrontare i metodi implementati.

Il report dovrà includere:

* descrizione degli algoritmi implementati;
* tabelle comparative;
* grafici e visualizzazioni;
* analisi statistiche.

Tra le metriche da considerare:

* tempo di esecuzione;
* percentuale di vittorie;
* numero di boss sconfitti;
* altre metriche significative emerse durante gli esperimenti.

Per fare questo creeremo una cartella con gli script per runnare i modelli e gli agenti necessari, ci saranno script con funzioni di plotting e script di analisi per comparare i modelli e creare questi report

---

## 5. README del progetto

Creare un README completo che includa:

* descrizione del progetto;
* struttura delle cartelle;
* istruzioni di installazione;
* dipendenze;
* come eseguire il progetto;
* esempi di comandi;
* descrizione delle principali funzionalità;
* spiegazione dei diversi agenti disponibili;
* eventuali esempi di utilizzo;
* sezione dedicata ai possibili sviluppi futuri e ai miglioramenti del progetto.

---

# Obiettivo finale

L'obiettivo è trasformare il progetto da un prototipo funzionante a un progetto pubblicabile, con codice pulito, documentazione completa, sistema di logging robusto, report degli esperimenti e README professionale.



A partire dal piano definito, crea un **piano di esecuzione dettagliato** organizzato come se fosse una board di progetto.

Per ogni attività:

* suddividila in **ticket indipendenti**;
* fai in modo che ogni ticket abbia uno scopo chiaro e verificabile;
* minimizza le dipendenze tra ticket, così da poterli svolgere in qualsiasi momento quando possibile;
* ordina i ticket secondo una sequenza logica che riduca il rischio di dover rifare lavoro già completato.

Per ogni ticket includi:

* titolo;
* obiettivo;
* contesto (perché esiste e quale problema risolve);
* attività da svolgere;
* criteri di completamento (Definition of Done);
* eventuali dipendenze da altri ticket;
* priorità;
* stima qualitativa dell'impegno (Piccolo / Medio / Grande).

crea una **skill riutilizzabile** che definisca un processo standard per trasformare qualsiasi progetto software completato in un progetto pronto per la pubblicazione.

La skill deve essere completamente generica e non fare riferimento al progetto attuale. Deve descrivere un workflow riutilizzabile che comprenda, ad esempio:

* analisi dello stato del progetto;
* individuazione delle attività di rifinitura;
* organizzazione del lavoro in ticket;
* definizione delle priorità;
* preparazione della documentazione;
* miglioramento della qualità del codice;
* predisposizione di test, logging e reportistica;
* preparazione del repository per la distribuzione.
