import pickle
import numpy as np
import torch
import torch.nn.functional
from dataset import TextDataset
from token_classifier import LSTMTokenClassifier


class LSTMSubstitutionsDetector:
    def __init__(self,
                 model: LSTMTokenClassifier,
                 word2id: dict,
                 device: str = 'cuda'):
        """
        :param model: LSTMTokenClassifier model that will be used for inference;
        :param word2id: mapping of words to id that was used for model training;
        :param device: one of ('cpu', 'cuda'), the device that will host computations;
        """

        self.model = model
        self.word2id = word2id
        self.device = device

    def run_model(self, line: str, unk_index: int = 1) -> np.ndarray:
        """
        Run self.model to obtain scores for the input line.

        :param line: input line;
        :param unk_index: index of the '<UNK>' tag in word2id;
        :return: probabilities of replacement for each word in line;
        """
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
        """
        Run inference on src_file. src_file is expected to contain one sequence per line
        where sequences are tokenized and tokens are space-separated.

        :param src_file: path to a source file;
        :param results_file: path to a file where the resulting scores will be written;
        """
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

    weights = '../weights/lstm_subst_detector_0009_0.4344.pt'  # F0.5 = 0.28
    model.load_state_dict(torch.load(weights, map_location=device))

    detector = LSTMSubstitutionsDetector(model, word2id, device)
    detector.run_inference_on_file(src_file='../data/val.src',
                                   results_file='../data/val.scores.lstm')
