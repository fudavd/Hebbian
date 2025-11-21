from src.maml import MetaMaml
from src.networks import ConvNet
from src.utils import TrainingLogger, get_dataset, split_dataset_by_class
from tqdm import tqdm
import torch
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True,)
    parser.add_argument("--shots", type=int, required=True)
    args = parser.parse_args()
    rep = args.rep
    shots = args.shots

    # Experiment configuration
    config = {
        'n_classes': 20,
        'n_shots': shots,
        'n_query': 15,
        'meta_epochs': 10000,
        'meta_learning_rate': 1e-3,
        'inner_learning_rate': 1e-2,
        'meta_batch_size': 32,
        'Model': [(28, 28), (64, 3), (64, 3), (64, 3), (64, 3)],
        'device': 'cuda',
        'dataset': 'Omniglot',
    }

    # Initialize logger
    logger = TrainingLogger(
        log_dir=F"./results/{config['dataset']}/N-{config['n_classes']}:K-{config['n_shots']}/{rep}",
        experiment_name=f"run_{rep}",
    )

    # Log experiment configuration
    logger.log_experiment_config(config)

    # Model setup
    model = ConvNet(
        layers=config['Model'],
        bias=True,
        device=config['device'],
    )
    meta_model = MetaMaml(
        model,
        n_classes=config['n_classes'],
        n_shots=config['n_shots'],
        lr_inner=config['inner_learning_rate'],
    )
    optimizer = torch.optim.Adam(
        list(meta_model.net.parameters()) + list(meta_model.head.parameters()),
        lr=config['meta_learning_rate']
    )

    train_dataset, test_dataset = get_dataset(config['dataset'])
    total_classes = len(train_dataset.classes)
    train_classes = max(total_classes-config['n_classes'], config['n_classes'])

    train_dataset, val_dataset = split_dataset_by_class(train_dataset, train_classes, config['n_classes'])

    logger.logger.info("Starting training...")

    # Training loop
    for meta_epoch in range(config['meta_epochs']):
        eps_test, _ = split_dataset_by_class(test_dataset, config['n_classes'], 0)
        eps_train, eps_val = split_dataset_by_class(train_dataset, config['n_classes'], config['n_classes'])

        optimizer.zero_grad()

        mean_acc = 0
        running_loss = 0

        # Meta-batch loop with progress bar
        with tqdm(range(config['meta_batch_size']),
                  desc=f"Epoch {meta_epoch}",
                  leave=False,
                  ncols=100) as pbar:

            meta_loss = 0
            for meta_batch in pbar:
                loss, accuracy = meta_model.forward(eps_train, eps_val)
                meta_loss += loss
                mean_acc += accuracy / config['meta_batch_size']
                running_loss += loss.item() / config['meta_batch_size']

                # Log meta-batch
                logger.log_meta_batch(meta_batch, loss.item(), accuracy)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy * 100:.2f}%'
                })
        meta_loss /= config['meta_batch_size']
        meta_loss.backward()
        optimizer.step()

        val_loss, val_accuracy = meta_model.validation(eps_test)
        # Log epoch metrics
        logger.log_epoch(meta_epoch, running_loss, mean_acc, val_loss, val_accuracy, optimizer, meta_model.net)

        # Print summary every epoch (optional, since logger already prints)
        print(f"\nEpoch {meta_epoch}: meta-loss {running_loss:.4f} | mean accuracy {mean_acc * 100:.2f}% | val accuracy {val_accuracy.item() * 100:.2f}%")

        # Save checkpoint every 100 epochs
        if meta_epoch % 1000 == 0:
            logger.save_checkpoint(meta_model, optimizer, meta_epoch)

    # Final save
    logger.save_checkpoint(meta_model, optimizer, config['meta_epochs'] - 1, "final_checkpoint.pt")
    logger.save_training_history()

    # Print final statistics
    training_time = logger.get_training_time()
    best_accuracy = max(logger.history['accuracy']) * 100
    logger.logger.info(f"Training completed in {training_time}")
    logger.logger.info(f"Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()