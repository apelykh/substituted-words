import os
import torch


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

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                lines, labels, line_lengths = batch
                # labels = labels.contiguous().view(-1)

                # predictions = self.model(lines.to(self.device))
                predictions = self.model(lines.to(self.device), line_lengths.int())
                # b = predictions.size(1)
                # t = predictions.size(0)
                # predictions = predictions.view(b * t, -1)

                loss = self.criterion(torch.squeeze(predictions).float(), labels.to(self.device).float())
                # loss = self.model.loss(predictions, labels.to(self.device), line_lengths.to(self.device))
                running_val_loss += loss.item()

        return running_val_loss / len(val_loader)

    def _train_epoch(self, train_loader):
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            # print(batch['train_features'].shape)
            self.optimizer.zero_grad()

            lines, labels, line_lengths = batch
            # labels = labels.contiguous().view(-1)

            # predictions = self.model(lines.to(self.device))
            predictions = self.model(lines.to(self.device), line_lengths.int())
            # b = predictions.size(1)
            # t = predictions.size(0)
            # predictions = predictions.view(b * t, -1)

            loss = self.criterion(torch.squeeze(predictions).float(), labels.to(self.device).float())

            # loss = self.model.loss(predictions, labels.to(self.device), line_lengths.to(self.device))
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
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
