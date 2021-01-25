import pickle
import numpy as np
import torch
import torch.nn.functional
from dataset import TextDataset
from lstm_token_classifier import LSTMTokenClassifier


class LSTMSubstitutionsDetector:
    def __init__(self,
                 model: LSTMTokenClassifier,
                 word2id: dict,
                 device: str = 'cuda'):

        self.model = model
        self.word2id = word2id
        self.device = device

    def run_model(self, line: str, unk_index=1) -> np.ndarray:
        encoded_line = np.array([self.word2id.get(word, unk_index)
                                 for word in TextDataset.preprocess_line(line)])
        line_tensor = torch.as_tensor(encoded_line[np.newaxis, :], dtype=torch.long)
        mask = np.ones(len(encoded_line))
        output = self.model(line_tensor.to(self.device),
                            torch.as_tensor(mask[np.newaxis, :], dtype=torch.int).to(self.device))

        scores = torch.softmax(output.logits, dim=2)
        scores = scores.detach().cpu().numpy()
        scores = scores[:, :, 1]

        return scores.squeeze()

    def run_inference_on_file(self, src_file: str, results_file: str):
        out = open(results_file, 'w')

        with open(src_file, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    print('Line {}'.format(i))

                scores = self.run_model(line)
                str_scores = ['{:.5f}'.format(elem) for elem in scores]
                out.write(' '.join(str_scores) + '\n')

        out.close()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('../assets/text_dataset_train_word2id_101945.pkl', 'rb') as f:
        word2id = pickle.load(f)

    model = LSTMTokenClassifier(len(word2id),
                                embedding_dim=100,
                                hidden_dim=200,
                                pretrained_embeddings=None).to(device)

    weights = '../weights/lstm_subst_detector_0009_0.4344.pt'  # F0.5 = 0.284
    model.load_state_dict(torch.load(weights, map_location=device))

    detector = LSTMSubstitutionsDetector(model, word2id, device)
    detector.run_inference_on_file(src_file='../data/val.src',
                                   results_file='../data/val.lstm.scores')
