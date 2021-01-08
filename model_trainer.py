import os
import torch
import numpy as np
# from seqeval.metrics import f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import get_linear_schedule_with_warmup


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, model_name='subst_detector'):
        self.weights_dir = './weights'
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = None
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = 1.0

    def _validate_epoch(self, val_loader):
        total_val_loss = 0.0
        predicted_labels = []
        true_labels = []

        for i, batch in enumerate(val_loader):
            batch = tuple(elem.to(self.device) for elem in batch)
            lines, labels, masks = batch

            with torch.no_grad():
                outputs = self.model(lines,
                                    token_type_ids=None,
                                    attention_mask=masks,
                                    labels=labels)

            total_val_loss += outputs[0].item()

            # (32, 200, 2)
            logits = outputs[1].detach().cpu().numpy()
            predictions = np.argmax(logits, axis=2)
            batch_size, seq_len = predictions.shape

            predicted_labels.extend(list(predictions.reshape(batch_size * seq_len).astype(np.int)))
            t_labels = labels.to('cpu').numpy()
            true_labels.extend(list(t_labels.reshape(batch_size * seq_len).astype(np.int)))

        print("Validation Accuracy: {}".format(accuracy_score(true_labels, predicted_labels)))
        print("Precision: {}".format(precision_score(true_labels, predicted_labels)))
        print("Recall: {}".format(recall_score(true_labels, predicted_labels)))
        print("Validation F1-Score: {}".format(f1_score(true_labels, predicted_labels)))

        return total_val_loss / len(val_loader)

    # -------------------------------------------------------------------------------------------
    #     # Put the model into evaluation mode
    #     model.eval()
    #     # Reset the validation loss for this epoch.
    #     eval_loss, eval_accuracy = 0, 0
    #     predictions, gt = [], []
    #
    #     for batch in valid_dataloader:
    #         batch = tuple(t.to(device) for t in batch)
    #         b_input_ids, b_input_mask, b_labels = batch
    #
    #         with torch.no_grad():
    #             outputs = model(b_input_ids, token_type_ids=None,
    #                             attention_mask=b_input_mask, labels=b_labels)
    #         # Move logits and labels to CPU
    #         logits = outputs[1].detach().cpu().numpy()
    #         label_ids = b_labels.to('cpu').numpy()
    #
    #         # Calculate the accuracy for this batch of test sentences.
    #         eval_loss += outputs[0].mean().item()
    #         predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    #         gt.extend(label_ids)
    #
    #     eval_loss = eval_loss / len(valid_dataloader)
    #     validation_loss_values.append(eval_loss)
    #     print("Validation loss: {}".format(eval_loss))
    #
    #     pred_tags = [tag_values[p_i] for p, l in zip(predictions, gt)
    #                  for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    #     valid_tags = [tag_values[l_i] for l in gt
    #                   for l_i in l if tag_values[l_i] != "PAD"]
    #     print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    #     print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    # -------------------------------------------------------------------------------------------

    def _train_epoch(self, train_loader):
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            lines, labels, masks = batch

            self.model.zero_grad()

            outputs = self.model(lines.to(self.device),
                                token_type_ids=None,
                                attention_mask=masks.to(self.device),
                                labels=labels.to(self.device))

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            every_n_batches = 30
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
            # self.model.train()
            # print('Epoch {}/{}'.format(epoch, start_epoch + num_epochs - 1))
            # avg_epoch_loss = self._train_epoch(train_loader)
            avg_epoch_loss = 0
            # train_loss.append(avg_epoch_loss)
            # print('Epoch {} train loss: {:.4f}'.format(epoch, avg_epoch_loss))

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
