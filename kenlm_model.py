import kenlm
import math


class KenLMModel:
    def __init__(self, model_path='./train.binary'):
        self.model = kenlm.LanguageModel(model_path)

    def __call__(self, seq):
        scores = [math.pow(10, prob) for prob, _, _ in self.model.full_scores(seq, bos=False, eos=False)]

        return scores


if __name__ == '__main__':
    sent = 'Library managed to both stay afloat and increase its piece of the shrunken market.'

    model = KenLMModel()
    probs = model(sent.rstrip().lower())
    print(probs)
    print(len(probs))
    print(len(sent.rstrip().lower().split(' ')))
