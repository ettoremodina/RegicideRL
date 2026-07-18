import time
import random
import argparse
from ml_logger import GameRecorder, get_logger, start_run
from ml_logger.serialization import serialize_game
from game.regicide import Game
from game.action_handler import ActionHandler
from solvers.parallel import ParallelSimulator
from agents.random_agent import RandomAgent
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper

logger = get_logger(__name__)


def simulate_normal(num_games=1000, recorder=None):
    logger.info("Starting normal single-thread benchmark")
    start_time = time.time()
    
    handler = ActionHandler(max_hand_size=8)
    victories = 0
    total_turns = 0
    enemies_defeated = 0
    
    for i in range(num_games):
        game = Game(num_players=1)
        if recorder:
            recorder.begin_game(game, metadata={"benchmark": "normal", "index": i})
        res = {}
        required_defense = 0
        
        while not game.game_over:
            state_before = serialize_game(game) if recorder else None
            current = game.current_player
            hand = game.get_player_hand(current)
            
            # Simple AI: randomly choose a valid action
            if required_defense > 0:
                actions = handler.get_all_possible_actions(hand, "defense", {'enemy_attack': required_defense})
                if not actions:
                    # Auto defeat if cannot defend
                    game.game_over = True
                    if recorder:
                        recorder.record_event(
                            {"kind": "no_defense", "phase": "defense"},
                            {"message": "No valid defense"},
                            game,
                            state_before,
                        )
                    break
                action = random.choice(actions)
                indices = handler.mask_to_card_indices(action, len(hand))
                res = game.defend_with_card_indices(indices)
                action_record = {
                    "kind": "defend",
                    "phase": "defense",
                    "card_indices": indices,
                    "cards": res.get("cards_discarded", []),
                }
                required_defense = res.get("defense_required", 0)
            else:
                state_info = {
                    'can_use_solo_jester': game.can_use_solo_jester(),
                    'enemy_attack': game.current_enemy.attack if game.current_enemy else 0
                }
                actions = handler.get_all_possible_actions(hand, "attack", state_info)
                if not actions:
                    game.game_over = True
                    if recorder:
                        recorder.record_event(
                            {"kind": "no_action", "phase": "attack"},
                            {"message": "No valid attack"},
                            game,
                            state_before,
                        )
                    break
                action = random.choice(actions)
                
                is_solo_jester = (len(action) == 9 and action[8] == 1)
                
                if is_solo_jester:
                    res = game.use_solo_jester("step1")
                    action_record = {"kind": "solo_jester", "phase": "attack"}
                else:
                    indices = handler.mask_to_card_indices(action, len(hand))
                    if handler.is_yield_action(action):
                        res = game.yield_turn()
                        action_record = {"kind": "yield", "phase": "attack"}
                    else:
                        res = game.play_card(indices)
                        action_record = {
                            "kind": "play",
                            "phase": "attack",
                            "card_indices": indices,
                            "cards": res.get("cards_played", []),
                        }
                
                required_defense = res.get("defense_required", 0)
                
                # Handle Jester choice (solo mode defaults back to player 1)
                if res.get("phase") == "next_player_choice":
                    game.choose_next_player(0)
            if recorder:
                recorder.record_event(action_record, res, game, state_before)
                    
            total_turns += 1
            
        if game.victory:
            victories += 1
        
        # 12 enemies total
        enemies_left = len(game.castle_deck) + (1 if game.current_enemy and not game.victory else 0)
        enemies_defeated += (12 - enemies_left)
        if recorder:
            recorder.finish(game)

    elapsed = time.time() - start_time
    fps = num_games / elapsed
    
    logger.info(
        "Normal benchmark: %d games, %.2fs, %.2f games/s, %.2f%% wins",
        num_games,
        elapsed,
        fps,
        victories / num_games * 100,
    )
    return fps

def simulate_parallel(num_games=1000, jobs=None, context=None):
    logger.info("Starting parallel benchmark with jobs=%s", jobs or "max")
    simulator = ParallelSimulator(n_jobs=jobs, run_context=context)
    
    metrics = simulator.run_eval(
        agent_cls=RandomAgent, 
        agent_kwargs={"name": "Random"}, 
        total_games=num_games
    )
    
    fps = metrics['games_per_second']
    logger.info(
        "Parallel benchmark: %d games, %.2fs, %.2f games/s, %.2f%% wins",
        num_games,
        metrics["total_time"],
        fps,
        metrics["win_rate"] * 100,
    )
    simulator.close()
    return fps

def simulate_training(device, steps=10000, recorder=None):
    logger.info("Starting training benchmark on %s", device.upper())
    import torch
    from sb3_contrib.ppo_mask import MaskablePPO
    
    raw_env = RegicideEnv(num_players=1, recorder=recorder)
    env = NumericObsWrapper(raw_env)
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=0,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
    )
    
    start_time = time.time()
    model.learn(total_timesteps=steps)
    end_time = time.time()
    
    elapsed = end_time - start_time
    fps = steps / elapsed
    
    logger.info(
        "Training benchmark: device=%s, steps=%d, elapsed=%.2fs, speed=%.2f steps/s",
        device.upper(),
        steps,
        elapsed,
        fps,
    )
    return fps

def simulate_env(num_games=1000, recorder=None):
    logger.info("Starting environment benchmark")
    start_time = time.time()
    
    env = RegicideEnv(num_players=1, recorder=recorder)
    victories = 0
    total_turns = 0
    
    for _ in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            # Random agent logic using action_mask
            action_mask = obs['action_mask']
            # Find valid actions
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if not valid_actions:
                break
            action = random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            total_turns += 1
            
        if env.game.victory:
            victories += 1

    elapsed = time.time() - start_time
    fps = num_games / elapsed
    
    logger.info(
        "Environment benchmark: %d games, %.2fs, %.2f games/s, %.2f%% wins",
        num_games,
        elapsed,
        fps,
        victories / num_games * 100,
    )
    return fps

def build_parser():
    parser = argparse.ArgumentParser(description="Regicide Benchmarking Utility")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "normal", "env", "parallel", "cpu", "gpu"],
        default="normal",
        help="Which benchmark to run (default: normal; use all for the full suite)",
    )
    parser.add_argument("--games", type=int, default=1000, 
                        help="Number of games to simulate for normal/env/parallel (default: 1000)")
    parser.add_argument("--steps", type=int, default=10000, 
                        help="Number of training steps to simulate for cpu/gpu (default: 10000)")
    parser.add_argument("--jobs", type=int, default=None, 
                        help="Number of workers for parallel benchmark (default: max cores)")
    
    return parser


def main():
    args = build_parser().parse_args()
    context = start_run("benchmark", config=vars(args))
    recorder = GameRecorder(context) if context.game_recording_enabled else None
    try:
        modes_to_run = select_modes(args.mode)
        results = run_benchmarks(args, modes_to_run, recorder, context)
        output = context.save_result("benchmark.json", results)
        context.complete({"result": str(output)})
    except Exception as error:
        context.fail(error)
        logger.exception("Benchmark failed")
        raise


def select_modes(requested_mode):
    if requested_mode == "all":
        modes_to_run = ["normal", "env", "parallel", "cpu"]
        import torch

        if torch.cuda.is_available():
            modes_to_run.append("gpu")
        else:
            logger.warning("CUDA is unavailable; skipping GPU benchmark")
    else:
        modes_to_run = [requested_mode]

    if requested_mode == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; cannot benchmark GPU")
    return modes_to_run


def run_benchmarks(args, modes_to_run, recorder, context):
    results = {}
    if "normal" in modes_to_run:
        results["Normal   (Games/sec)"] = simulate_normal(args.games, recorder)
    if "env" in modes_to_run:
        results["Env      (Games/sec)"] = simulate_env(args.games, recorder)
    if "parallel" in modes_to_run:
        results["Parallel (Games/sec)"] = simulate_parallel(
            args.games,
            args.jobs,
            context,
        )
    if "cpu" in modes_to_run:
        results["Train CPU (Steps/sec)"] = simulate_training(
            "cpu",
            args.steps,
            recorder,
        )
    if "gpu" in modes_to_run:
        results["Train GPU (Steps/sec)"] = simulate_training(
            "cuda",
            args.steps,
            recorder,
        )
    if len(results) > 1:
        logger.info("Benchmark summary")
        for name, fps in results.items():
            logger.info("%s: %.2f", name, fps)
    return results

if __name__ == "__main__":
    main()
