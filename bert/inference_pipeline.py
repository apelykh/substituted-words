import torch
from dataset import BertTextDataset
from transformers import TokenClassificationPipeline, BertTokenizer, BertForTokenClassification


class BertSubstitutionsDetector:
    """
    An inference abstraction that combines a trained model, tokenizer and a pipeline for token
    classification to run inference on files.
    """
    def __init__(self,
                 model: BertForTokenClassification,
                 tokenizer: BertTokenizer):

        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = TokenClassificationPipeline(model, tokenizer, device=device)

    @staticmethod
    def is_subtoken(word: str) -> bool:
        """ Return True if word is a sub-token, internally created by BertTokenizer"""
        return True if word[:2] == "##" else False

    def join_subtokens(self, tokens: list, scores: list) -> tuple:
        """
        Join subword tokens produced by BertTokenizer into full words while synchronizing scores.

        :param tokens: sequence of tokens produced by BertTokenizer;
        :param scores: sequence of scores (one for each token);
        :return: lists of full words and corresponding word scores;
        """
        joined_tokens = []
        joined_scores = []

        for i in range(len(tokens)):
            if not self.is_subtoken(tokens[i]):
                joined_tokens.append(tokens[i])
                joined_scores.append(scores[i])
            else:
                joined_tokens[-1] = joined_tokens[-1] + tokens[i][2:]
                joined_scores[-1] = (joined_scores[-1] + scores[i]) / 2

        return joined_tokens, joined_scores

    def restore_words_with_scores(self, orig_tokens: list, tokens: list, scores: list) -> tuple:
        """
        Fix a destructive behavior of BertTokenizer input preprocessing to ensure that the length
        of the model outputs is the same as the length of the input string (in number of tokens).

        Examples of destructive behavior:
            "..." -> [".", ".", "."]
            "capital's" -> ["capital", "'", "s"]
            "anti-government" -> ["anti", "-", "government"]

        :param orig_tokens: sequence of tokens obtained by splitting the original line with spaces;
        :param tokens: sequence of tokens produced by BertTokenizer;
        :param scores: sequence of scores (one for each token) produced by the model;
        :return: restored tokens and their corresponding scores;
        """
        tokens, scores = self.join_subtokens(tokens, scores)

        i = 0
        j = 0
        inside_word = False
        restored_text = []
        joined_scores = []

        while i < len(orig_tokens):
            if not inside_word:
                buffer = orig_tokens[i]

            if tokens[j] == orig_tokens[i]:
                restored_text.append(tokens[j])
                joined_scores.append(scores[j])
                i += 1
                j += 1
            elif buffer.startswith(tokens[j]):
                if len(buffer) == len(orig_tokens[i]):
                    restored_text.append(tokens[j])
                    joined_scores.append(scores[j])
                    inside_word = True
                elif len(buffer) < len(orig_tokens[i]):
                    restored_text[-1] += tokens[j]
                    joined_scores[-1] = (joined_scores[-1] + scores[j]) / 2
                else:
                    raise RuntimeError

                buffer = buffer[len(tokens[j]):]
                j += 1
            else:
                raise RuntimeError

            if len(buffer) == 0:
                inside_word = False
                i += 1

        return restored_text, joined_scores

    @staticmethod
    def prepare_lines(src_file: str, max_line_len: int) -> list:
        """
        Separate each line from src_file into segments of length max_line_len.

        :return: list of segments for each line in the source file;
        """
        lines = []

        with open(src_file, 'r') as f:
            for i, line in enumerate(f):
                filtered_tokens = BertTextDataset.preprocess_line(line)

                lines.append([filtered_tokens[l_i * max_line_len: (l_i + 1) * max_line_len]
                              for l_i in range(int(len(filtered_tokens) / max_line_len) + 1)])
        return lines

    def run_inference_on_file(self, src_file: str, results_file: str, max_line_len: int = 100):
        """
        Run inference pipeline on src_file. src_file is expected to contain one sequence per line
        where sequences are tokenized and tokens are space-separated.

        :param src_file: path to a source file;
        :param results_file: path to a file where the resulting scores will be written;
        :param max_line_len: long lines will be separated into segments of length max_line_len
            before passing to the model;
        """
        lines = self.prepare_lines(src_file, max_line_len)

        with open(results_file, 'w') as out:
            for i, line_chunks in enumerate(lines):
                line_scores = []

                if i % 1000 == 0:
                    print('Line {}'.format(i))

                for line_chunk_tokens in line_chunks:
                    line_str = ' '.join(line_chunk_tokens)
                    outputs = self.pipeline(line_str)

                    tokens = [outputs[i]['word'] for i in range(len(outputs))]
                    scores = [1 - outputs[i]['score'] if outputs[i]['entity'] == 'LABEL_0' else outputs[i]['score']
                              for i in range(len(outputs))]

                    merged_tokens, merged_scores = self.restore_words_with_scores(line_chunk_tokens, tokens, scores)

                    line_scores.extend(merged_scores)

                str_scores = ['{:.5f}'.format(score) for score in line_scores]
                out.write(' '.join(str_scores) + '\n')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BertForTokenClassification.\
        from_pretrained('../assets/bert_subst_detector_0004_0.0491').to(device)

    # model = BertForTokenClassification.from_pretrained(
    #     "bert-base-uncased",
    #     # 2 labels -> logits: (32, 200, 2)
    #     num_labels=2,
    #     output_attentions=False,
    #     output_hidden_states=False
    # ).to(device)

    # weights = '../weights/run3_after_bugfix/subst_detector_0004_0.0491.pt'  # F0.5 = 0.84
    # model.load_state_dict(torch.load(weights, map_location=device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    detector = BertSubstitutionsDetector(model, tokenizer)
    detector.run_inference_on_file(
        '../data/val.src',
        '../data/val.scores.bert'
    )
