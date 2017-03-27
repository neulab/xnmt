
class Logger:

    REPORT_TEMPLATE = 'Epoch %.4f: {}_ppl=%.4f (loss/word=%.4f, words=%d)'

    def __init__(self, eval_every):
        self.sent_num = 0
        self.sent_num_not_report = 0
        self.eval_every = eval_every
        pass

    def count_tgt_words(self):
        raise NotImplementedError('count_tgt_words must be implemented in Logger subclasses')

    def count_sent_num(self):
        raise NotImplementedError('count_tgt_words must be implemented in Logger subclasses')

    def clear_counters(self):
        self.sent_num = 0
        self.sent_num_not_report = 0

    def report_ppl(self):
        pass


class BatchLogger(Logger):

    count_tgt_words = lambda tgt_words: sum(len(x) for x in tgt_words)

    count_sent_num = lambda x: len(x)


class NonBatchLogger(Logger):

    count_tgt_words = lambda tgt_words: len(tgt_words)

    count_sent_num = lambda x: 1