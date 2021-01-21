import unicodedata
import torch.nn.functional
from transformers import TokenClassificationPipeline, BertTokenizer, BertForTokenClassification

device = 'cuda'


def is_subtoken(word):
    return True if word[:2] == "##" else False


def join_subtokens(tokens, scores):
    joined_tokens = []
    joined_scores = []

    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]):
            joined_tokens.append(tokens[i])
            joined_scores.append(scores[i])
        else:
            joined_tokens[-1] = joined_tokens[-1] + tokens[i][2:]
            joined_scores[-1] = (joined_scores[-1] + scores[i]) / 2

    # TODO: remove ugliness
    assert len(joined_scores) == len(joined_tokens)

    return joined_tokens, joined_scores


def detokenize_with_scores(orig_tokens, tokens, scores):
    tokens, scores = join_subtokens(tokens, scores)

    i = 0
    j = 0
    inside_word = False
    restored_text = []
    joined_scores = []

    while i < len(orig_tokens):
        if not inside_word:
            t = orig_tokens[i]

        if tokens[j] == orig_tokens[i]:
            restored_text.append(tokens[j])
            joined_scores.append(scores[j])
            i += 1
            j += 1
        elif t.startswith(tokens[j]):
            if len(t) == len(orig_tokens[i]):
                restored_text.append(tokens[j])
                joined_scores.append(scores[j])
                inside_word = True
            elif len(t) < len(orig_tokens[i]):
                restored_text[-1] += tokens[j]
                joined_scores[-1] = (joined_scores[-1] + scores[j]) / 2
            else:
                raise NotImplementedError

            t = t[len(tokens[j]):]
            j += 1
        else:
            print(orig_tokens)
            print(tokens)
            raise ValueError

        if len(t) == 0:
            inside_word = False
            i += 1

    return restored_text, joined_scores


def run_inference_on_file(src_file, results_file, pipeline, max_line_len=100):
    lines = []

    with open(src_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split(' ')

            filtered_tokens = []
            for token in line:
                filtered_token = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('ascii').lower()
                if filtered_token == '':
                    filtered_token = '[UNK]'
                filtered_tokens.append(filtered_token)

            assert (len(filtered_tokens) == len(line))

            lines.append([filtered_tokens[l_i * max_line_len: (l_i + 1) * max_line_len]
                          for l_i in range(int(len(filtered_tokens) / max_line_len) + 1)])

    with open(results_file, 'w') as out:
        for i, line_chunks in enumerate(lines):
            line_scores = []

            if i % 1000 == 0:
                print('Line {}'.format(i))

            for line_chunk_tokens in line_chunks:
                line_str = ' '.join(line_chunk_tokens)
                outputs = pipeline(line_str)

                tokens = [outputs[i]['word'] for i in range(len(outputs))]
                scores = [1 - outputs[i]['score'] if outputs[i]['entity'] == 'LABEL_0' else outputs[i]['score']
                          for i in range(len(outputs))]

                merged_tokens, merged_scores = detokenize_with_scores(line_chunk_tokens, tokens, scores)

                assert(len(line_chunk_tokens) == len(merged_tokens))

                line_scores.extend(merged_scores)

            str_scores = ['{:.5f}'.format(score) for score in line_scores]
            out.write(' '.join(str_scores) + '\n')


if __name__ == '__main__':
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        # 2 labels -> logits: (32, 200, 2)
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)

    weights = './weights/run3_after_bugfix/subst_detector_0002_0.0542.pt'
    model.load_state_dict(torch.load(weights, map_location=device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    pipeline = TokenClassificationPipeline(model, tokenizer, device=0)
    run_inference_on_file('./data/val.src', './data/val.bert.scores', pipeline)
