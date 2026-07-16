"""
AlphaZero-style Expert Iteration module for Regicide.

Implements Phase 5 of the expert iteration plan: a synchronous training loop
where ISMCTS (with PUCT selection and network leaf evaluation) acts as the
"Expert" and a dual-headed neural network acts as the "Apprentice".
"""
