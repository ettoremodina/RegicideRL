# Agents package

def __getattr__(name):
    if name == 'BaseAgent':
        from .base_agent import BaseAgent
        return BaseAgent
    elif name == 'RandomAgent':
        from .random_agent import RandomAgent
        return RandomAgent
    elif name == 'HeuristicAgent':
        from .heuristic_agent import HeuristicAgent
        return HeuristicAgent
    elif name == 'PPOAgent':
        from .ppo_agent import PPOAgent
        return PPOAgent
    elif name == 'PIMCAgent':
        from .pimc_agent import PIMCAgent
        return PIMCAgent
    elif name == 'ISMCTSAgent':
        from .ismcts_agent import ISMCTSAgent
        return ISMCTSAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['BaseAgent', 'RandomAgent', 'HeuristicAgent', 'PPOAgent', 'PIMCAgent', 'ISMCTSAgent']
