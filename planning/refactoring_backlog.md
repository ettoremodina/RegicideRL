# Refactoring backlog

Questo backlog separa le attività sicure già applicate dagli interventi che
richiedono test di regressione dedicati. L'obiettivo è migliorare la struttura
senza modificare le regole di Regicide o il comportamento degli agenti.

## R-01 — Spezzare `ActionHandler`

- Obiettivo: separare generazione delle combinazioni, validazione della
  sopravvivenza e serializzazione dell'action space.
- Contesto: `get_global_action_mask` e `get_all_possible_actions` sono ancora
  funzioni lunghe e contengono logica parallela.
- Attività: estrarre generatori condivisi; aggiungere test parametrizzati per
  attacco, difesa, immunità dei semi e Jester; mantenere l'API attuale.
- Definition of Done: nessuna duplicazione tra maschere locali e globali,
  test completi verdi, benchmark non peggiorato.
- Dipendenze: R-04.
- Priorità: alta. Stima: grande.

## R-02 — Separare il ciclo di gioco da `Game`

- Obiettivo: ridurre la responsabilità della classe `Game` senza cambiare il
  modello di stato.
- Contesto: setup, turni, attacco, difesa, pesca e transizioni sono concentrati
  nello stesso modulo.
- Attività: estrarre servizi interni solo dopo aver definito test di stato e
  invarianti; mantenere `game.regicide` come facciata compatibile.
- Definition of Done: test delle regole invariati, clone deterministico e API
  pubblica compatibile.
- Dipendenze: nessuna, ma richiede copertura aggiuntiva.
- Priorità: media. Stima: grande.

## R-03 — Ridurre la complessità delle pipeline RL/MCTS

- Obiettivo: separare orchestrazione, valutazione, persistenza e reporting.
- Contesto: MCTS, self-play, trainer, probe e orchestrator contengono funzioni
  lunghe e dipendenze opzionali.
- Attività: introdurre oggetti di configurazione per le fasi, sostituire i
  percorsi hardcoded con `Path`, aggiungere smoke test con rete minima.
- Definition of Done: import senza side effect, smoke test CPU, output degli
  esperimenti compatibile.
- Dipendenze: R-01.
- Priorità: media. Stima: grande.

## R-04 — Standardizzare test e action-space contract

- Obiettivo: rendere espliciti dimensioni, offset e conversioni dell'action
  space.
- Contesto: il progetto aveva riferimenti obsoleti a 542 azioni mentre il
  contratto corrente è di 543 azioni.
- Attività: usare `game.action_space` nei nuovi test e nei solver; aggiungere
  test round-trip per ogni categoria di azione.
- Definition of Done: nessun numero magico nei moduli runtime e test di
  conversione completi.
- Dipendenze: nessuna.
- Priorità: alta. Stima: media.

