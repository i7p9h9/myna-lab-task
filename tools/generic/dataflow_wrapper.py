import dataflow as D


def dataflow_wrapper(generator, workers=1, num_prefetch=1):
    class DFromGenerator(D.DataFlow):
        def __init__(self, generator, *args, **kwargs):
            self.generator = generator
            self.n = 0
            self.ans = self.generator[0]
            super(DFromGenerator, self).__init__(*args, **kwargs)

        def __iter__(self):
            return self

        def __next__(self):
            return self.generator[self.n]

        def __call__(self):
            yield self.ans
            self.ans = self.generator[0]

    class DGenerator():
        def __init__(self):
            self.d = D.MultiProcessRunner(DFromGenerator(generator=generator),
                                          num_prefetch=num_prefetch,
                                          num_proc=workers)
            self.d.reset_state()

        def __getattr__(self, name):
            return generator.__getattribute__(name)

        def __len__(self):
            return len(generator)

        def __getitem__(self, item):
            return next(self.d.get_data())

    return DGenerator()