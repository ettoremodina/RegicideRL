1. I log fanno schifo
2. Logga in un file tutte le mie richieste di cambiamento su questo argomento (descrivi nel file md in questione), lo scopo è avere le spec pronte, per ricordare cosa voglio e cosa mi serve
3. non è possibile usare tecnologie più adatte? Non ne so molto di GUI ma non pensavo python fosse la scelta migliore. Io adesso sto ancora capendo cosa voglio inserire in questa dashboard, quindi voglio qualcosa di customizzabile, inoltre voglio che sia adattabile a progetti futuri, non solo regicide
4. non posso chiudere la schermata se non scrivo qualcosa in certe box
5. mostra il comando che sta lanciando quando seleziono qualcosa da runnare
6. config panel per le config nell'app
7. documentazione nel control panel



LIVE OUTPUT

Build comparison report
failed · 1s · PID 27248

×
Control Panel job: Build comparison report
Started: 2026-07-19T23:47:43.074465+00:00
Source: scripts/experimental_report/orchestrator.py
Command: C:\Users\modin\miniconda3\python.exe -m scripts.experimental_report.orchestrator --config C:\Users\modin\Desktop\programming\GAMES\Regicide\artifacts\control_panel\jobs\panel-20260719T234743-05eceb0f\config-snapshots\config_test.yaml --agents random heuristic --games 10 --base-seed 42 --jobs 1
--------------------------------------------------------------------------------
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\modin\Desktop\programming\GAMES\Regicide\scripts\experimental_report\orchestrator.py", line 141, in <module>
    main()
    ~~~~^^
  File "C:\Users\modin\Desktop\programming\GAMES\Regicide\scripts\experimental_report\orchestrator.py", line 130, in main
    run_pipeline(
    ~~~~~~~~~~~~^
        config_path=arguments.config,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        jobs=arguments.jobs,
        ^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\modin\Desktop\programming\GAMES\Regicide\scripts\experimental_report\orchestrator.py", line 33, in run_pipeline
    context, report_config, effective_config_path, is_resuming = _prepare_run(
                                                                 ~~~~~~~~~~~~^
        config_path,
        ^^^^^^^^^^^^
    ...<3 lines>...
        resume_run,
        ^^^^^^^^^^^
    )
    ^
  File "C:\Users\modin\Desktop\programming\GAMES\Regicide\scripts\experimental_report\orchestrator.py", line 83, in _prepare_run
    load_report_config(config_path),
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "C:\Users\modin\Desktop\programming\GAMES\Regicide\scripts\experimental_report\configuration.py", line 20, in load_report_config
    raise ValueError("Missing 'experimental_report' section in config")
ValueError: Missing 'experimental_report' section in config