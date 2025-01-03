class CoresetMethod(object):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, **kwargs):
        # if fraction <= 0.0 or fraction > 1.0:
            # raise ValueError("Illegal Coreset Size.")

        self.dst_train = dst_train
        self.num_classes = args.n_class #len(dst_train.classes)
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)
        if fraction > 1.0:
            self._fraction = 0.1
            self.fraction = int(self.fraction)
        self.coreset_size = round(self.n_train * self._fraction)

    def select(self, **kwargs):
        return

