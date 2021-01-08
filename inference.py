import numpy as np
import torch
import torch.nn.functional
from dataset import TextDataset
from model import WordSubstitutionDetector
from kenlm_model import KenLMModel
from utils import create_embedding_matrix

device = 'cuda'


def run_model(model, model_state, device, line, word2id, unk_index=1):
    with torch.no_grad():
        encoded_line = np.array([word2id.get(word, unk_index) for word in TextDataset.preprocess_line(line)])
        line_tensor = torch.as_tensor(encoded_line[np.newaxis, :], dtype=torch.long)
        # line_len = torch.as_tensor([np.newaxis, :], dtype=torch.int64)
        # word_probs = model(line_tensor.to(device), [len(encoded_line)])
        word_probs, model_state = model(line_tensor.to(device), [len(encoded_line)], model_state)
        word_probs = torch.nn.functional.softmax(torch.squeeze(word_probs))

        word_probs = word_probs.detach().cpu().numpy()
        # probability scores of each real word from LM
        inv_scores = word_probs.take(encoded_line)

    return 1 - inv_scores


def run_kenlm_model(model, line):
    line = line.rstrip().lower()
    probs = model(line)

    return [1 - elem for elem in probs]


def run_inference_on_file(src_file, results_file, model, word2id):
    out = open(results_file, 'w')
    model_state = model.init_states(init_method='zeros')

    with open(src_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 500 == 0:
                print('Line {}'.format(i))

            scores = run_model(model, model_state, device, line, word2id)
            str_scores = ['{:.5f}'.format(elem) for elem in scores]
            out.write(' '.join(str_scores) + '\n')

    out.close()


if __name__ == '__main__':
    # TODO: log word2id during train time and load here?
    train_dataset = TextDataset(base_path='./data',
                                split_name='train_small',
                                max_len=300,
                                freq_threshold=15)

    word2id = train_dataset.word2id
    pretrained_embeddings = create_embedding_matrix('../glove.6B/glove.6B.100d.txt', word2id, len(word2id), 100)

    # TODO: serialize model with vocab and word2id
    model = WordSubstitutionDetector(len(word2id),
                                     embedding_dim=100,
                                     hidden_dim=200,
                                     pretrained_embeddings=pretrained_embeddings).to(device)

    # weights = './weights/mixmodel_subst_detector_1s_0029_0.0526.pt'
    weights = './weights/subst_detector_0001_0.1949.pt'
    model.load_state_dict(torch.load(weights, map_location=device))

    src_file = './data/dev.src'
    run_inference_on_file(src_file, './data/dev.scores', model, word2id)
