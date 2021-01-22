import os
import torch
import numpy as np
from eval import _f_measure
from sklearn.metrics import accuracy_score, precision_score, recall_score


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, model_name='subst_detector'):
        self.weights_dir = '../weights'
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _validate_epoch(self, val_loader):
        total_val_loss = 0.0
        predicted_labels = []
        true_labels = []

        for i, batch in enumerate(val_loader):
            lines, labels, line_lengths = batch
            # labels = labels.contiguous().view(-1)

            with torch.no_grad():
                predictions = self.model(lines.to(self.device),
                                         line_lengths.int())

            predictions = torch.squeeze(predictions)
            loss = self.criterion(predictions.float(),
                                  labels.to(self.device).float())
            total_val_loss += loss.item()

            scores = torch.sigmoid(predictions)
            scores = (scores.cpu().numpy() > 0.5).astype(np.int)
            batch_size, seq_len = scores.shape
            a = list(scores.reshape(batch_size * seq_len))
            predicted_labels.extend(a)

            b = list(labels.numpy().reshape(batch_size * seq_len).astype(np.int))
            true_labels.extend(b)

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
            self.optimizer.zero_grad()

            lines, labels, line_lengths = batch
            # labels = labels.contiguous().view(-1)

            predictions = self.model(lines.to(self.device), line_lengths.int())
            # b = predictions.size(1)
            # t = predictions.size(0)
            # predictions = predictions.view(b * t, -1)

            loss = self.criterion(torch.squeeze(predictions).float(),
                                  labels.to(self.device).float())
            loss.backward()
            self.optimizer.step()

            every_n_batches = 30
            if i % every_n_batches == every_n_batches - 1:
                print('[%d/%4d] loss: %.3f' % (i + 1, len(train_loader), loss.item()))

            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, train_loader, val_loader, start_epoch, num_epochs):
        train_loss = []
        val_loss = []

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.model.train()
            print('Epoch {}/{}'.format(epoch, start_epoch + num_epochs - 1))
            avg_epoch_loss = self._train_epoch(train_loader)
            train_loss.append(avg_epoch_loss)
            print('Epoch {} train loss: {:.4f}'.format(epoch, avg_epoch_loss))

            # ---------------------------------------------------------------
            # validation
            self.model.eval()
            avg_epoch_val_loss = self._validate_epoch(val_loader)
            val_loss.append(avg_epoch_val_loss)
            print('Epoch {} val loss: {:.4f}'.format(epoch, avg_epoch_val_loss))
            print('-' * 50)

            weights_file = os.path.join(self.weights_dir,
                                        '{}_{:04d}_{:.4f}.pt'.format(self.model_name, epoch, avg_epoch_loss))
            torch.save(self.model.state_dict(), weights_file)

        return train_loss, val_loss
