import os
import torch
from torch.autograd import Variable


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, model_name='subst_detector'):
        self.weights_dir = './weights'
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _validate_epoch(self, val_loader):
        running_val_loss = 0.0
        state_h, state_c = self.model.init_states(init_method='zeros')

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                lines, targets, line_lengths = batch
                # targets = targets.contiguous().view(-1)

                predictions, (state_h, state_c) = self.model(lines.to(self.device),
                                                             line_lengths.int(),
                                                             (state_h, state_c))
                # b = predictions.size(1)
                # t = predictions.size(0)
                # predictions = predictions.view(b * t, -1)

                predictions = predictions.permute(0, 2, 1)

                loss = self.criterion(predictions.float(), targets.to(self.device))
                running_val_loss += loss.item()

        return running_val_loss / len(val_loader)

    def _train_epoch(self, train_loader):
        with torch.autograd.set_detect_anomaly(True):
            running_loss = 0.0
            state_h, state_c = self.model.init_states(init_method='zeros')

            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                lines, targets, line_lengths = batch
                # targets = targets.contiguous().view(-1)

                predictions, (state_h, state_c) = self.model(lines.to(self.device),
                                                             line_lengths.int(),
                                                             (state_h, state_c))
                # b = predictions.size(1)
                # t = predictions.size(0)
                # predictions = predictions.view(b * t, -1)

                state_h = state_h.detach()
                state_c = state_c.detach()

                # [batch_size, seq_len, num_classes] -> [batch_size, num_classes, seq_len]
                predictions = predictions.permute(0, 2, 1)

                loss = self.criterion(predictions.float(), targets.to(self.device))
                loss.backward()
                self.optimizer.step()

                every_n_batches = 30
                if i % every_n_batches == every_n_batches - 1:
                    print('[%d/%4d] loss: %.3f' % (i + 1, len(train_loader), loss.item()))

                running_loss += loss.item()
        return running_loss / len(train_loader)

    def fit(self, train_loader, val_loader, start_epoch, num_epochs):
        train_loss = []
        val_loss = []

        for epoch in range(start_epoch, start_epoch + num_epochs):
            print('Epoch {}/{}'.format(epoch, start_epoch + num_epochs - 1))
            avg_epoch_loss = self._train_epoch(train_loader)
            train_loss.append(avg_epoch_loss)
            print('Epoch {} train loss: {:.4f}'.format(epoch, avg_epoch_loss))

            # ---------------------------------------------------------------
            # validation
            avg_epoch_val_loss = self._validate_epoch(val_loader)
            val_loss.append(avg_epoch_val_loss)
            print('Epoch {} val loss: {:.4f}'.format(epoch, avg_epoch_val_loss))
            print('-' * 50)

            weights_file = os.path.join(self.weights_dir,
                                        '{}_{:04d}_{:.4f}.pt'.format(self.model_name, epoch, avg_epoch_loss))
            torch.save(self.model.state_dict(), weights_file)

        return train_loss, val_loss
