import copy
import os
import importlib
import pickle
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report

from evaluate.metric import calculate_competition_score
from utils.config import load_yaml, validate_experiment_config, replace_config_with_args
from utils.logger import setup_logger, config_logging
from data.fm import FMDataLoader
from data.dataset import create_torch_data_loader, set_random_seeds
from utils.parse_args import parse_args
from utils.loss import get_loss_function, compute_focal_loss_alpha
from utils.metric import compute_metrics
from utils.plot import plot_metric_at_k
from utils.optim import create_parameter_groups

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/fm.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


def main() -> None:
    args = parse_args()
    # set result path
    if not args.result_path:
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        test_flag = "test" if args.is_test else "untest"
        result_path = RESULT_PATH.format(test=test_flag, model=args.model, dt=dt)
        if args.postfix is not None:
            result_path = f"{result_path}-{args.postfix}"
    else:
        result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    # load config
    config = load_yaml(CONFIG_PATH)
    # replace config used in this experiment
    config = replace_config_with_args(args, config, result_path)
    # set up logger
    logger = setup_logger(os.path.join(result_path, "log.log"))

    # dictionary to store various statistics
    metric = {
        "train": {
            "loss": [],
            "auc": [],
            "f1": [],
            "recall": [],
            "precision": [],
        },
        "val": {
            "loss": [],
            "auc": [],
            "f1": [],
            "recall": [],
            "precision": [],
        },
    }
    metric_at_best_cpt_score = {
        "auc": 0.0,
        "macro_f1": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "competition_score": 0.0,
        "ap": 0.0,
        "wll": 0.0,
        "epoch": 0,
    }

    try:
        # log experiment configuration
        config_logging(logger, args, config, result_path)

        # validate experiment configuration
        validate_experiment_config(args, config)

        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load dataset
        data_loader = FMDataLoader(config)
        train_x, train_y, train_seq, val_x, val_y, val_seq = (
            data_loader.load_train_data(is_fm=True, is_test=args.is_test)
        )
        test_x, test_ids, test_seq = data_loader.load_test_data(is_fm=True)
        logger.info(
            f"Split completed -> number of train / val : {train_x.size(0)} / {val_x.size(0)}"
        )
        logger.info(f"Number of test samples: {test_x.size(0)}")

        assert train_x.shape[1] == val_x.shape[1] == test_x.shape[1]

        logger.info(f"Number of numerical features: {len(config.data.num_features)}")
        logger.info(f"Number of categorical features: {len(config.data.cat_features)}")

        # setup torch dataloader
        set_random_seeds(config.data.seed)
        train_dataloader, val_dataloader, test_dataloader = create_torch_data_loader(
            train_x=train_x,
            train_y=train_y,
            train_seq=train_seq,
            val_x=val_x,
            val_y=val_y,
            val_seq=val_seq,
            test_x=test_x,
            test_seq=test_seq,
            batch_size=config.common.batch_size,
            test_only=False,
            num_numeric_feat=len(config.data.num_features),
            num_cat_feat=len(config.data.cat_features),
            num_workers=config.common.num_workers,
            stratified_sampling=True,
            positive_ratio=0.02,  # ratio of positive samples
            seed=config.data.seed,
            use_seq_feature=args.use_seq_feature,
            max_seq_length=config.data.max_seq_length,
            recent_ratio=config.data.recent_ratio,
        )

        # setup model
        model_path = (
            f"models.fm.{args.model}.{args.model}_seq"
            if args.use_seq_feature
            else f"models.fm.{args.model}.{args.model}"
        )
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            categorical_field_dims=data_loader.categorical_field_dims,
            numerical_field_count=len(config.data.num_features),
            vocab_size=config.data.max_id_in_seq + 1,
            embed_dim=config.common.embedding_dim,
            mlp_dims=config.model.deepfm.mlp_dims,
            cin_layer_dims=config.model.deepfm.cin_layer_dims,
            reduction_ratio=config.model.fibinet.reduction_ratio,
            cross_layers=config.model.dcn.num_cross_layers
            if args.model == "dcn"
            else config.model.dcn_v2.num_cross_layers,
            deep_layers=config.model.dcn.deep_layers
            if args.model == "dcn"
            else config.model.dcn_v2.deep_layers,
            structure=config.model.dcn_v2.structure,
            low_rank=config.model.dcn_v2.low_rank,
            num_experts=config.model.dcn_v2.num_experts,
            max_seq_length=config.data.max_seq_length,
            d_model=config.model.seq.d_model,
            nhead=config.model.seq.nhead,
            use_causal_mask=config.model.seq.use_causal_mask,
        ).to(device)

        # start training!
        param_groups = create_parameter_groups(
            model,
            base_lr=config.common.learning_rate,
            fm_lr_multiplier=config.optim.fm_lr_multiplier,
            deep_lr_multiplier=config.optim.deep_lr_multiplier,
            cin_lr_multiplier=config.optim.cin_lr_multiplier,
        )
        optimizer = optim.Adam(param_groups, weight_decay=config.common.regularization)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.common.epochs, eta_min=config.optim.lr_eta_min
        )
        criterion = get_loss_function(
            config.loss.name,
            alpha=compute_focal_loss_alpha(train_y),
            pos_weight=config.loss.bce.pos_weight,
            focal_gamma=config.loss.focal.gamma,
            pos_lr_multiplier=config.loss.focal.pos_lr_multiplier,
            neg_lr_multiplier=config.loss.focal.neg_lr_multiplier,
        )
        if config.loss.name == "focal":
            logger.info(f"Focal Loss Alpha: {compute_focal_loss_alpha(train_y):.4f}")
        best_cpt_score = -float("inf")
        early_stopping = False

        logger.info("Starting training...")
        # train model
        for epoch in range(config.common.epochs):
            logger.info(f"################## epoch {epoch} ##################")

            # Set epoch for stratified sampler (important for proper shuffling)
            if hasattr(train_dataloader, "batch_sampler") and hasattr(
                train_dataloader.batch_sampler, "set_epoch"
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)
            train_loss = 0.0
            val_loss = 0.0
            train_pred_proba = []
            train_true = []
            every_train_epoch = len(train_dataloader) // 6
            every_val_epoch = len(val_dataloader) // 6
            model.train()
            for batch_idx, batch_data in enumerate(train_dataloader):
                if len(batch_data) == 4:
                    num_features, cat_features, seq, labels = batch_data
                    seq = seq.to(device)
                else:
                    num_features, cat_features, labels = batch_data
                    seq = None
                num_features = num_features.to(device)
                cat_features = cat_features.to(device)
                labels = labels.to(device)
                pred = model(num_features, cat_features, seq=seq)
                loss = criterion(pred, labels)

                # Check for NaN loss before backprop
                if torch.isnan(loss):
                    logger.info(
                        f"NaN loss detected at batch {batch_idx}, epoch {epoch}. Skipping this batch."
                    )
                    continue

                # backpropagation
                optimizer.zero_grad()
                loss.backward()

                # Check for NaN gradients
                nan_detected = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.info(
                            f"NaN gradient detected in {name} at batch {batch_idx}"
                        )
                        optimizer.zero_grad()
                        nan_detected = True
                        break

                if nan_detected:
                    logger.info(
                        f"NaN gradient detected at batch {batch_idx}, epoch {epoch}. Skipping this batch."
                    )
                    continue  # Skip this batch

                optimizer.step()

                train_loss += loss

                # Collect predictions and true labels for metrics calculation
                pred_probs = torch.sigmoid(pred).squeeze(1).cpu().detach().numpy()
                train_pred_proba.extend(pred_probs)
                train_true.extend(labels.squeeze(1).cpu().numpy())

                if batch_idx % every_train_epoch == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_dataloader)}")

                # Gradient clipping to prevent explosion
                # Add gradient norm monitoring
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                if batch_idx % every_train_epoch == 0:
                    logger.info(f"Batch {batch_idx}: Gradient norm = {total_norm:.4f}")

            scheduler.step()
            # Print current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"LR: {current_lr:.7f}")

            train_loss /= len(train_dataloader)
            logger.info(f"train loss: {round(train_loss.item(), 5)}")
            metric["train"]["loss"].append(train_loss.item())

            train_metric = compute_metrics(train_true, train_pred_proba)

            metric["train"]["auc"].append(train_metric["auc"])
            metric["train"]["f1"].append(train_metric["macro_f1"])
            metric["train"]["recall"].append(train_metric["recall"])
            metric["train"]["precision"].append(train_metric["precision"])

            model.eval()
            val_pred_proba = []
            val_true = []

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(val_dataloader):
                    if len(batch_data) == 4:
                        num_features, cat_features, seq, labels = batch_data
                        seq = seq.to(device)
                    else:
                        num_features, cat_features, labels = batch_data
                        seq = None
                    # Set epoch for stratified sampler (important for proper shuffling)
                    if hasattr(val_dataloader, "batch_sampler") and hasattr(
                        val_dataloader.batch_sampler, "set_epoch"
                    ):
                        val_dataloader.batch_sampler.set_epoch(epoch)
                    num_features = num_features.to(device)
                    cat_features = cat_features.to(device)
                    labels = labels.to(device)
                    pred = model(num_features, cat_features, seq=seq)
                    loss = criterion(pred, labels)

                    val_loss += loss

                    # Collect predictions and true labels for metrics calculation
                    pred_probs = torch.sigmoid(pred).squeeze(1).cpu().numpy()
                    pred_probs = np.clip(pred_probs, 1e-15, 1 - 1e-15)
                    val_pred_proba.extend(pred_probs)
                    val_true.extend(labels.squeeze(1).cpu().numpy())

                    if batch_idx % every_val_epoch == 0:
                        logger.info(f"Batch {batch_idx}/{len(val_dataloader)}")
                val_loss /= len(val_dataloader)
                logger.info(f"val loss: {round(val_loss.item(), 5)}")
                metric["val"]["loss"].append(val_loss.item())

            val_metric = compute_metrics(val_true, val_pred_proba)

            # Calculate AUC score
            logger.info(f"val AUC: {round(val_metric['auc'], 5)}")
            logger.info(f"val macro F1: {round(val_metric['macro_f1'], 5)}")
            logger.info(f"val recall: {round(val_metric['recall'], 5)}")
            logger.info(f"val precision: {round(val_metric['precision'], 5)}")

            # Generate classification report (convert probabilities to binary predictions)
            val_pred_binary = [1 if pred >= 0.5 else 0 for pred in val_pred_proba]
            logger.info(f"\n{classification_report(val_true, val_pred_binary)}")

            metric["val"]["auc"].append(val_metric["auc"])
            metric["val"]["f1"].append(val_metric["macro_f1"])
            metric["val"]["recall"].append(val_metric["recall"])
            metric["val"]["precision"].append(val_metric["precision"])

            # calculate competition score
            cpt_score, ap, wll = calculate_competition_score(
                np.array(val_true), np.array(val_pred_proba), True
            )

            logger.info(f"val competition score: {cpt_score}")
            logger.info(f"val ap: {ap}")
            logger.info(f"val wll: {wll}")

            # early stopping logic
            if best_cpt_score < cpt_score:
                prev_best_cpt_score = best_cpt_score
                best_cpt_score = cpt_score
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = config.common.patience
                logger.info(
                    f"Best validation competition score: {round(best_cpt_score, 5)}, Previous validation competition score: {round(prev_best_cpt_score, 5)}"
                )

                # store metrics at best competition score
                metric_at_best_cpt_score["auc"] = val_metric["auc"]
                metric_at_best_cpt_score["macro_f1"] = val_metric["macro_f1"]
                metric_at_best_cpt_score["precision"] = val_metric["precision"]
                metric_at_best_cpt_score["recall"] = val_metric["recall"]

                metric_at_best_cpt_score["competition_score"] = cpt_score
                metric_at_best_cpt_score["ap"] = ap
                metric_at_best_cpt_score["wll"] = wll

                metric_at_best_cpt_score["epoch"] = epoch
            else:
                patience -= 1
                logger.info(
                    f"Validation competition score did not decrease. Patience {patience} left."
                )
                if patience == 0:
                    logger.info(
                        f"Patience over. Early stopping at epoch {epoch} with {round(best_cpt_score, 5)} validation competition score"
                    )
                    early_stopping = True

            if early_stopping is True:
                break

        # Load the best model weights
        model.load_state_dict(best_model_weights)
        logger.info("Load weight with best validation competition score")

        # ensure that best model weights are saved
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "categorical_field_dims": data_loader.categorical_field_dims,
                    "numerical_field_count": len(config.data.num_features),
                    "vocab_size": config.data.max_id_in_seq + 1,
                    "embed_dim": config.common.embedding_dim,
                    "mlp_dims": config.model.deepfm.mlp_dims,
                    "cin_layer_dims": config.model.deepfm.cin_layer_dims,
                    "reduction_ratio": config.model.fibinet.reduction_ratio,
                    "cross_layers": config.model.dcn.num_cross_layers
                    if args.model == "dcn"
                    else config.model.dcn_v2.num_cross_layers,
                    "deep_layers": config.model.dcn.deep_layers
                    if args.model == "dcn"
                    else config.model.dcn_v2.deep_layers,
                    "structure": config.model.dcn_v2.structure,
                    "low_rank": config.model.dcn_v2.low_rank,
                    "num_experts": config.model.dcn_v2.num_experts,
                    "max_seq_length": config.data.max_seq_length,
                    "d_model": config.model.seq.d_model,
                    "nhead": config.model.seq.nhead,
                    "use_causal_mask": config.model.seq.use_causal_mask,
                },
            },
            os.path.join(result_path, "model_weight.pt"),
        )
        # save metric stored at each epoch
        pickle.dump(metric, open(os.path.join(result_path, "metric.pkl"), "wb"))

        # plot metric
        plot_metric_at_k(metric, parent_save_path=result_path)

        # log metrics at best competition score
        logger.info(
            f"############### Metrics at best competition score (epoch {metric_at_best_cpt_score['epoch']}) ###############"
        )
        for metric, value in metric_at_best_cpt_score.items():
            logger.info(f"{metric} = {value}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
