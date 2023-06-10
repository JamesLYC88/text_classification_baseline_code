import logging
import os

import numpy as np
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import AutoTokenizer

from libmultilabel.nn import data_utils
from libmultilabel.nn.model import Model
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer
from libmultilabel.common_utils import dump_log


class TorchTrainer:
    """A wrapper for training neural network models with pytorch lightning trainer.

    Args:
        config (AttributeDict): Config of the experiment.
        datasets (dict, optional): Datasets for training, validation, and test. Defaults to None.
        classes(list, optional): List of class names.
        word_dict(torchtext.vocab.Vocab, optional): A vocab object which maps tokens to indices.
        embed_vecs (torch.Tensor, optional): The pre-trained word vectors of shape (vocab_size, embed_dim).
        search_params (bool, optional): Enable pytorch-lightning trainer to report the results to ray tune
            on validation end during hyperparameter search. Defaults to False.
        save_checkpoints (bool, optional): Whether to save the last and the best checkpoint or not.
            Defaults to True.
    """
    def __init__(
        self,
        config: dict,
        datasets: dict = None,
        classes: list = None,
        word_dict: dict = None,
        embed_vecs = None,
        search_params: bool = False,
        save_checkpoints: bool = True
    ):
        self.run_name = config.run_name
        self.checkpoint_dir = config.checkpoint_dir
        self.log_path = config.log_path
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set up seed & device
        if not config.enable_transformer_trainer:
            from libmultilabel.nn.nn_utils import set_seed
            set_seed(seed=config.seed)
        self.device = init_device(use_cpu=config.cpu)
        self.config = config

        # Load pretrained tokenizer for dataset loader
        self.tokenizer = None
        tokenize_text = 'lm_weight' not in config.network_config
        if not tokenize_text:
            self.tokenizer = AutoTokenizer.from_pretrained(config.network_config['lm_weight'], use_fast=True)
        # Load dataset
        if datasets is None:
            self.datasets = data_utils.load_datasets(
                train_path=config.train_path,
                test_path=config.test_path,
                val_path=config.val_path,
                val_size=config.val_size,
                merge_train_val=config.merge_train_val,
                tokenize_text=tokenize_text,
                remove_no_label_data=config.remove_no_label_data
            )
        else:
            self.datasets = datasets

        self._setup_model(classes=classes,
                          word_dict=word_dict,
                          embed_vecs=embed_vecs,
                          log_path=self.log_path,
                          checkpoint_path=config.checkpoint_path)
        if config.enable_transformer_trainer:
            from transformers import EarlyStoppingCallback, set_seed, Trainer

            from libmultilabel.nn.data_utils import generate_transformer_batch
            from libmultilabel.nn.nn_utils import init_training_args, compute_metrics

            training_args = init_training_args(config)

            set_seed(training_args.seed)

            self.train_dataset = self._get_dataset_loader(split='train', shuffle=config.shuffle).dataset
            self.val_dataset = self._get_dataset_loader(split='val').dataset
            self.test_dataset = self._get_dataset_loader(split='test').dataset
            self.trainer = Trainer(
                model=self.model.network.lm,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                compute_metrics=compute_metrics,
                tokenizer=self.tokenizer,
                data_collator=generate_transformer_batch,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)]
            )
        else:
            self.trainer = init_trainer(checkpoint_dir=self.checkpoint_dir,
                                        epochs=config.epochs,
                                        patience=config.patience,
                                        val_metric=config.val_metric,
                                        silent=config.silent,
                                        use_cpu=config.cpu,
                                        limit_train_batches=config.limit_train_batches,
                                        limit_val_batches=config.limit_val_batches,
                                        limit_test_batches=config.limit_test_batches,
                                        search_params=search_params,
                                        save_checkpoints=save_checkpoints,
                                        accumulate_grad_batches=config.accumulate_grad_batches)
            callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, ModelCheckpoint)]
            self.checkpoint_callback = callbacks[0] if callbacks else None

        # Dump config to log
        dump_log(self.log_path, config=config)

    def _setup_model(
        self,
        classes: list = None,
        word_dict: dict = None,
        embed_vecs = None,
        log_path: str = None,
        checkpoint_path: str = None
    ):
        """Setup model from checkpoint if a checkpoint path is passed in or specified in the config.
        Otherwise, initialize model from scratch.

        Args:
            classes(list): List of class names.
            word_dict(torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
            embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
            log_path (str): Path to the log file. The log file contains the validation
                results for each epoch and the test results. If the `log_path` is None, no performance
                results will be logged.
            checkpoint_path (str): The checkpoint to warm-up with.
        """
        if 'checkpoint_path' in self.config and self.config.checkpoint_path is not None:
            checkpoint_path = self.config.checkpoint_path

        if checkpoint_path is not None:
            logging.info(f'Loading model from `{checkpoint_path}`...')
            self.model = Model.load_from_checkpoint(checkpoint_path)
        else:
            logging.info('Initialize model from scratch.')
            if self.config.embed_file is not None:
                logging.info('Load word dictionary ')
                word_dict, embed_vecs = data_utils.load_or_build_text_dict(
                    dataset=self.datasets['train'],
                    vocab_file=self.config.vocab_file,
                    min_vocab_freq=self.config.min_vocab_freq,
                    embed_file=self.config.embed_file,
                    silent=self.config.silent,
                    normalize_embed=self.config.normalize_embed,
                    embed_cache_dir=self.config.embed_cache_dir
                )
            if not classes:
                classes = data_utils.load_or_build_label(
                    self.datasets, self.config.label_file, self.config.include_test_labels)

            if self.config.val_metric not in self.config.monitor_metrics:
                logging.warn(
                    f'{self.config.val_metric} is not in `monitor_metrics`. Add {self.config.val_metric} to `monitor_metrics`.')
                self.config.monitor_metrics += [self.config.val_metric]

            self.model = init_model(model_name=self.config.model_name,
                                    network_config=dict(self.config.network_config),
                                    classes=classes,
                                    word_dict=word_dict,
                                    embed_vecs=embed_vecs,
                                    init_weight=self.config.init_weight,
                                    log_path=log_path,
                                    learning_rate=self.config.learning_rate,
                                    optimizer=self.config.optimizer,
                                    momentum=self.config.momentum,
                                    weight_decay=self.config.weight_decay,
                                    metric_threshold=self.config.metric_threshold,
                                    monitor_metrics=self.config.monitor_metrics,
                                    silent=self.config.silent,
                                    save_k_predictions=self.config.save_k_predictions,
                                    zero=self.config.zero,
                                    multi_class=self.config.multi_class,
                                    enable_ce_loss=self.config.enable_ce_loss,
                                    hierarchical=self.config.hierarchical
                                   )

    def _get_dataset_loader(self, split, shuffle=False):
        """Get dataset loader.

        Args:
            split (str): One of 'train', 'test', or 'val'.
            shuffle (bool): Whether to shuffle training data before each epoch. Defaults to False.

        Returns:
            torch.utils.data.DataLoader: Dataloader for the train, test, or valid dataset.
        """
        return data_utils.get_dataset_loader(
            data=self.datasets[split],
            word_dict=self.model.word_dict,
            classes=self.model.classes,
            device=self.device,
            max_seq_length=self.config.max_seq_length,
            batch_size=self.config.batch_size if split == 'train' else self.config.eval_batch_size,
            shuffle=shuffle,
            data_workers=self.config.data_workers,
            tokenizer=self.tokenizer,
            add_special_tokens=self.config.add_special_tokens,
            hierarchical=self.config.hierarchical,
            enable_transformer_trainer=self.config.enable_transformer_trainer,
            multi_class=self.config.multi_class
        )

    def train(self):
        """Train model with pytorch lightning trainer. Set model to the best model after the training
        process is finished.
        """
        assert self.trainer is not None, "Please make sure the trainer is successfully initialized by `self._setup_trainer()`."
        if self.config.enable_transformer_trainer:
            # Training
            train_result = self.trainer.train()
            metrics = train_result.metrics
            metrics['train_samples'] = len(self.train_dataset)
            self.trainer.save_model()
            self.trainer.log_metrics('train', metrics)
            self.trainer.save_metrics('train', metrics)
            self.trainer.save_state()
            return

        train_loader = self._get_dataset_loader(split='train', shuffle=self.config.shuffle)

        if 'val' not in self.datasets:
            logging.info('No validation dataset is provided. Train without vaildation.')
            self.trainer.fit(self.model, train_loader)
        else:
            from pytorch_lightning.callbacks.early_stopping import EarlyStopping
            self.trainer.callbacks += [EarlyStopping(patience=self.config.patience,
                                                    monitor=self.config.val_metric,
                                                    mode='max')]  # tentative hard code
            val_loader = self._get_dataset_loader(split='val')
            self.trainer.fit(self.model, train_loader, val_loader)

        # Set model to the best model. If the validation process is skipped during
        # training (i.e., val_size=0), the model is set to the last model.
        model_path = self.checkpoint_callback.best_model_path or self.checkpoint_callback.last_model_path
        if model_path:
            logging.info(f'Finished training. Load best model from {model_path}.')
            self._setup_model(checkpoint_path=model_path)
        else:
            logging.info('No model is saved during training. \
                If you want to save the best and the last model, please set `save_checkpoints` to True.')

    def test(self, split='test'):
        """Test model with pytorch lightning trainer. Top-k predictions are saved
        if `save_k_predictions` > 0.

        Args:
            split (str, optional): One of 'train', 'test', or 'val'. Defaults to 'test'.

        Returns:
            dict: Scores for all metrics in the dictionary format.
        """
        assert 'test' in self.datasets and self.trainer is not None

        if self.config.enable_transformer_trainer:
            # Validation
            metrics = self.trainer.evaluate(eval_dataset=self.val_dataset)
            metrics['val_samples'] = len(self.val_dataset)
            self.trainer.log_metrics('val', metrics)
            self.trainer.save_metrics('val', metrics)
            # Testing
            predictions, labels, metrics = self.trainer.predict(self.test_dataset, metric_key_prefix='test')
            metrics['test_samples'] = len(self.test_dataset)
            self.trainer.log_metrics('test', metrics)
            self.trainer.save_metrics('test', metrics)
            return

        logging.info(f'Testing on {split} set.')
        test_loader = self._get_dataset_loader(split=split)
        metric_dict = self.trainer.test(self.model, dataloaders=test_loader)[0]

        if self.config.save_k_predictions > 0:
            self._save_predictions(test_loader, self.config.predict_out_path)

        return metric_dict

    def _save_predictions(self, dataloader, predict_out_path):
        """Save top k label results.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the test or valid dataset.
            predict_out_path (str): Path to the an output file holding top k label results.
        """
        batch_predictions = self.trainer.predict(self.model, dataloaders=dataloader)
        pred_labels = np.vstack([batch['top_k_pred']
                                for batch in batch_predictions])
        pred_scores = np.vstack([batch['top_k_pred_scores']
                                for batch in batch_predictions])
        with open(predict_out_path, 'w') as fp:
            for pred_label, pred_score in zip(pred_labels, pred_scores):
                out_str = ' '.join([f'{self.model.classes[label]}:{score:.4}' for label, score in zip(
                    pred_label, pred_score)])
                fp.write(out_str+'\n')
        logging.info(f'Saved predictions to: {predict_out_path}')
