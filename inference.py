import numpy as np
import torch
import torch.nn.functional
from dataset import TextDataset
from model import WordSubstitutionDetector

device = 'cuda'


def run_model(model, device, line, word2id, unk_index=1):
    encoded_line = np.array([word2id.get(word, unk_index) for word in TextDataset.preprocess_line(line)])
    line_tensor = torch.as_tensor(encoded_line[np.newaxis, :], dtype=torch.long)
    # line_len = torch.as_tensor([np.newaxis, :], dtype=torch.int64)
    scores = model(line_tensor.to(device), [len(encoded_line)])

    return torch.nn.functional.sigmoid(scores)


def run_inference_on_file(src_file, results_file, model, word2id):
    out = open(results_file, 'w')

    with open(src_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 500 == 0:
                print('Line {}'.format(i))

            tokens = line.split(' ')
            scores = run_model(model, device, tokens, word2id)
            str_scores = ['{:.5f}'.format(elem) for elem in scores.squeeze().detach().cpu().numpy()]
            out.write(' '.join(str_scores) + '\n')

    out.close()


if __name__ == '__main__':
    # TODO: log word2id during train time and load here?
    train_dataset = TextDataset(base_path='./data', split_name='train_small', max_len=None)
    vocab = train_dataset.get_vocab()

    # TODO: serialize model with vocab and word2id
    model = WordSubstitutionDetector(len(vocab) + 2, embedding_dim=100, hidden_dim=200).to(device)
    # weights = './weights/mixmodel_subst_detector_1s_0029_0.0526.pt'
    weights = './weights/subst_detector_0029_0.0332.pt'
    model.load_state_dict(torch.load(weights, map_location=device))

    src_file = './data/train_small.src'
    run_inference_on_file(src_file, './data/train_small.scores', model, train_dataset.word2id)
