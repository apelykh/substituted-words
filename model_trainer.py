import os
import torch
import numpy as np
from eval import _f_measure
from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup


class GeneralModelTrainer:
    def __init__(self, model, criterion, optimizer, device, model_prefix='bert_subst_detector'):
        self.weights_dir = './weights'
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        self.model_prefix = model_prefix
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = None
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = 1.0
        self.num_labels = 2

    def _validate_epoch(self, val_loader):
        total_val_loss = 0.0
        predicted_labels = []
        true_labels = []

        for i, batch in enumerate(val_loader):
            batch = tuple(elem.to(self.device) for elem in batch)
            lines, labels, masks = batch

            with torch.no_grad():
                outputs = self.model(lines, attention_mask=masks)
            #                                      labels=labels)

            active_loss = masks.view(-1) == 1

            active_logits = outputs.logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(self.criterion.ignore_index).type_as(labels)
            )
            loss = self.criterion(active_logits, active_labels)
            total_val_loss += loss.item()
            #             total_val_loss += outputs.loss.item()

            # (32, 200, 2)
            logits = outputs.logits.detach().cpu().numpy()
            predictions = np.argmax(logits, axis=2)
            batch_size, seq_len = predictions.shape

            predicted_labels.extend(list(predictions.reshape(batch_size * seq_len).astype(np.int)))
            t_labels = labels.to('cpu').numpy()
            true_labels.extend(list(t_labels.reshape(batch_size * seq_len).astype(np.int)))

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)

        print("Validation Accuracy: {}".format(accuracy_score(true_labels, predicted_labels)))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("Validation F0.5-Score: {}".format(_f_measure(precision, recall, 0.5)))

        return total_val_loss / len(val_loader)

    def _train_epoch(self, train_loader):
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            batch = tuple(elem.to(self.device) for elem in batch)
            lines, labels, masks = batch
            # lines = lines.to(self.device)
            # labels = labels.to(self.device)
            # masks = masks.to(self.device)

            # self.model.zero_grad()
            self.optimizer.zero_grad()

            outputs = self.model(lines, attention_mask=masks)
            #                                  labels=labels)

            #             loss = outputs[0]

            active_loss = masks.view(-1) == 1

            active_logits = outputs.logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(self.criterion.ignore_index).type_as(labels)
            )
            loss = self.criterion(active_logits, active_labels)

            total_loss += loss.item()
            loss.backward()

            every_n_batches = 100
            if i % every_n_batches == every_n_batches - 1:
                print('[%d/%4d] loss: %.3f' % (i + 1, len(train_loader), loss.item()))

            # Clip the norm of the gradient to help prevent the exploding gradients problem.
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                           max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()

        return total_loss / len(train_loader)

    def fit(self, train_loader, val_loader, start_epoch, num_epochs):
        train_loss = []
        val_loss = []

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * num_epochs
        )

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.model.train()
            print('Epoch {}/{}'.format(epoch, start_epoch + num_epochs - 1))
            avg_epoch_loss = self._train_epoch(train_loader)
            train_loss.append(avg_epoch_loss)
            print('Epoch {} train loss: {:.4f}'.format(epoch, avg_epoch_loss))

            self.model.eval()
            avg_epoch_val_loss = self._validate_epoch(val_loader)
            val_loss.append(avg_epoch_val_loss)
            print('Epoch {} val loss: {:.4f}'.format(epoch, avg_epoch_val_loss))
            print('-' * 50)

            weights_file = os.path.join(self.weights_dir,
                                        '{}_{:04d}_{:.4f}.pt'.format(self.model_prefix, epoch, avg_epoch_loss))
            torch.save(self.model.state_dict(), weights_file)

        return train_loss, val_loss
