## 4. Agenti e partita simulata

Esegue una partita completa con logging:

```powershell
python -m scripts.log_game
```

Analizza i log prodotti:

```powershell
python -m scripts.analyze_runs
```

Testa il solver/evaluator con un carico ridotto:

```powershell
python -m solvers.train --agent random --episodes 1 --games_per_episode 200 --jobs 1
```

Con `--jobs 1` il simulatore usa direttamente il processo corrente, evitando
il costo di avvio e comunicazione del multiprocessing.

## 5. UI Pygame

Avvia l’interfaccia grafica:

```powershell
python -m ui
```

Durante la difesa, se nessuna combinazione è sufficiente, la partita termina
oppure consente di usare il Solo Jester quando disponibile.

## 7. Training PPO

Training ridotto usando la configurazione di test:

```powershell
python -m solvers.train_rl --config config_test.yaml
```

Training completo usando `config.yaml`:

```powershell
python -m solvers.train_rl --config config.yaml
```

`invalid_action_rate` misura le azioni rifiutate dal motore. Con le maschere
corrette deve restare a zero: gli yield illegali e le difese impossibili non
sono più presentati alla policy.

Questi comandi possono creare file in `models/` e log TensorBoard in `runs/`.

## 8. Behavioral Cloning

Genera un dataset minimo:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python -m solvers.generate_bc_data --games 2 --out outputs/bc_data_smoke.npz
```

Esegue un’epoca di pre-training:

```powershell
python -m solvers.bc_train --data outputs/bc_data_smoke.npz --epochs 1 --batch 8
```

Il modello pre-trained viene salvato in `models/ppo_bc_pretrained.zip`.

## 9. Hyperparameter tuning

Controllo del CLI:

```powershell
python -m solvers.tune --help
```

Tuning ridotto con `config_test.yaml`:

```powershell
python -m solvers.tune --config config_test.yaml
```

Il tuning completo può richiedere molto tempo e produce log TensorBoard e
modelli temporanei in `runs/`.

## 10. Analisi di un modello addestrato

Controllo del CLI:

```powershell
python -m solvers.analysis.run_analysis --help
```

Analisi reale, da eseguire dopo aver prodotto un modello PPO:

```powershell
python -m solvers.analysis.run_analysis `
  --model models/ppo_regicide_tuned.zip `
  --games 2
```

Se necessario, specifica anche il log TensorBoard:

```powershell
python -m solvers.analysis.run_analysis `
  --model models/ppo_regicide_tuned.zip `
  --games 2 `
  --logdir runs/rl_logs
```

## 12. AlphaZero

Avvia AlphaZero tramite l’entry point unificato:

```powershell
python -m solvers.train --agent alphazero --config config.yaml
```

Smoke test:

```powershell
python -m solvers.train --agent alphazero `
  --config config_test.yaml `
  --az-iterations 1 `
  --games-per-episode 1 `
  --sims 1 `
  --eval-games 1
```

Usa `python -m solvers.train --help` per gli override e `--resume` per
riprendere un checkpoint.
