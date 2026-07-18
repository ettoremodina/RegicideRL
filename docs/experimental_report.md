# Regicide: Analisi Sperimentale ISMCTS

## Introduzione
Questo documento sintetizza i risultati dell'agente ISMCTS (Information Set Monte Carlo Tree Search) applicato al gioco di carte cooperativo Regicide (modalità solitario).

## Metodologia
L'agente utilizza l'algoritmo ISMCTS per gestire l'informazione nascosta intrinseca nel mazzo. 
Ad ogni turno, l'agente:
1. **Determinizza** lo stato nascosto campionando configurazioni valide del mazzo rimanente.
2. Esegue rollout a partire da questi stati.
3. Utilizza un'euristica basata sui danni attesi e la difesa richiesta per guidare la simulazione.
4. Seleziona l'azione con la frequenza di visita maggiore nel root node.

## Risultati
Puoi generare le metriche aggiornate sulle run effettuate lanciando lo script:
```bash
python scripts/analyze_runs.py
```

Le metriche di interesse includono:
* **Win Rate**: Percentuale di partite vinte dall'agente.
* **Avg Bosses Defeated**: Numero medio di boss (su 12) sconfitti per run.
* **Performance e Tempo**: Tempo di esecuzione per mossa.

## Conclusioni e Sviluppi Futuri
L'utilizzo dell'algoritmo ISMCTS permette di raggiungere uno stato di vittoria superando la sfida complessa posta dall'informazione nascosta. Sviluppi futuri potrebbero riguardare l'ottimizzazione in C++ delle fasi di rollout per estendere il tempo di ricerca, o l'utilizzo combinato di Reti Neurali per ridurre lo spazio di ricerca (AlphaZero-style).
