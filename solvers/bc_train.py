"""Behavioral-cloning pre-training with canonical run artifacts."""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sb3_contrib.ppo_mask import MaskablePPO
from torch.utils.data import DataLoader, TensorDataset

from ml_logger import RunContext, get_logger, start_run
from solvers.architecture import RegicideFeatureExtractor
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper

logger = get_logger(__name__)


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def train_bc(data_path, context: RunContext, config_path="config.yaml", epochs=10, batch_size=64):
    logger.info("Loading behavioral-cloning data from %s", data_path)
    data = np.load(data_path)
    dataloader = _build_dataloader(data, batch_size)
    config = load_config(config_path)
    model = _build_model(config)
    optimizer = optim.Adam(model.policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    metrics = []
    for epoch in range(epochs):
        epoch_metrics = _train_epoch(model, dataloader, optimizer, criterion)
        epoch_metrics["epoch"] = epoch + 1
        metrics.append(epoch_metrics)
        context.log_metrics(epoch + 1, epoch_metrics)
        logger.info(
            "Epoch %d/%d: loss=%.4f accuracy=%.2f%%",
            epoch + 1,
            epochs,
            epoch_metrics["loss"],
            epoch_metrics["accuracy"] * 100,
        )
    output_path = context.run_dir / "models" / "ppo_bc_pretrained"
    model.save(output_path)
    logger.info("Saved pre-trained model to %s.zip", output_path)
    return {"model": str(output_path) + ".zip", "metrics": metrics}


def _build_dataloader(data, batch_size):
    tensors = (
        torch.tensor(data["hand_values"], dtype=torch.long),
        torch.tensor(data["hand_suits"], dtype=torch.long),
        torch.tensor(data["enemy_stats"], dtype=torch.float32),
        torch.tensor(data["flags"], dtype=torch.float32),
        torch.tensor(data["action_masks"], dtype=torch.float32),
        torch.tensor(data["actions"], dtype=torch.long),
    )
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)


def _build_model(config):
    ppo_config = config["ppo"]
    architecture = ppo_config.get("net_arch", [256, 256])
    policy_kwargs = {
        "features_extractor_class": RegicideFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": architecture, "vf": architecture},
    }
    environment = NumericObsWrapper(RegicideEnv(num_players=1))
    return MaskablePPO(
        "MultiInputPolicy",
        environment,
        device=ppo_config["device"],
        verbose=0,
        policy_kwargs=policy_kwargs,
    )


def _train_epoch(model, dataloader, optimizer, criterion):
    model.policy.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        hand_values, hand_suits, enemy, flags, masks, actions = batch
        observations = {
            "hand_values": hand_values.to(model.device),
            "hand_suits": hand_suits.to(model.device),
            "enemy_stats": enemy.to(model.device),
            "flags": flags.to(model.device),
            "action_mask": masks.to(model.device),
        }
        expected_actions = actions.to(model.device).flatten()
        features = model.policy.extract_features(observations)
        latent_policy = model.policy.mlp_extractor.forward_actor(features)
        logits = model.policy.action_net(latent_policy)
        loss = criterion(logits, expected_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (torch.argmax(logits, dim=1) == expected_actions).sum().item()
        total += expected_actions.size(0)
    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    context = start_run("bc-training", config=vars(args))
    try:
        result = train_bc(args.data, context, args.config, args.epochs, args.batch)
        context.save_result("training.json", result)
        context.complete({"model": result["model"]})
    except Exception as error:
        context.fail(error)
        logger.exception("Behavioral-cloning training failed")
        raise


if __name__ == "__main__":
    main()
