import numpy as np
import num2words

import typing

from tqdm import tqdm as tqdm_base


class NumEncoder:
    """
    So funny class, I am the first time working with this task - it was difficult 
    for me to create this class, and it turned out so ugly, as possible. 
    I believe, it's possible to create it in a more attarctive way, moreover, 
    I guess one exists in open source, but I didn't found.
    """
    def __init__(self):
        self.range_ones = [num2words.num2words(n, lang="ru")[:-1] for n in np.arange(1, 11)]
        self.range_teenagers = [num2words.num2words(n, lang="ru") for n in np.arange(11, 20)]
        self.range_adult = [num2words.num2words(n, lang="ru") for n in np.arange(20, 100, 10)]
        self.range_old = [num2words.num2words(n, lang="ru") for n in np.arange(100, 1000, 100)]
        
        self.num2class = None
        self.class2num = None
        self.ending_teenagers = None
        self.ending_adults = None
        self.ending_old = None
        
        self.ugly_init()
        
    def __len__(self):
        return len(self.num2class)
    
    @staticmethod
    def process_female(text: str) -> str:
        return text.replace("две", "два").replace("одна", "один")
    
    def ugly_init(self):
        """
        OMG
        """
        ending_teenagers = list()
        for n in range(9):
            ending_teenagers.append(self.range_teenagers[n].replace("две", "два").split(self.range_ones[n])[-1])
        self.ending_teenagers = set(ending_teenagers)
        
        ending_adults = list()
        for n in range(8):
            ending_adults.append(self.range_adult[n].replace("две", "два").split(self.range_ones[n + 1])[-1])
        self.ending_adults = set(ending_adults)
        
        ending_old = list()
        for n in range(9):
            ending_old.append(self.range_old[n].replace("две", "два").split(self.range_ones[n])[-1])
        self.ending_old = set(ending_old)
        
        self.num2class = {x: n for n, x in enumerate(self.range_ones)}
        self.num2class["десять"] = 10
        self.num2class["сорок"] = 11
        self.num2class["девяносто"] = 12
        self.num2class["ноль"] = 13
        self.num2class["тысяч"] = 14

        self.num2class["teen"] = 15
        self.num2class["adult"] = 16
        self.num2class["old"] = 17
        self.num2class["space"] = 18
        
        # reversed
        self.class2num = {_num: _class for _num, _class in enumerate(np.arange(1, 11))}
        self.class2num[10] = 10
        self.class2num[11] = 40
        self.class2num[12] = 90
        self.class2num[13] = 0

        self.class2num[14] = 1000
        self.class2num[15] = 10
        self.class2num[16] = 10
        self.class2num[17] = 100
        self.class2num[18] = "space"
        
    def encode(self, num: int) -> typing.List[int]:
        assert 0 <= num < 1000_000, "out of range, [0, 1000.000) allowed"
        
        text = num2words.num2words(num, lang="ru")
        words = self.process_female(text).split(" ")

        t = list()
        for m, w in enumerate(words):
            t.append(18)  # space at start
            if w in self.num2class:
                t.append(self.num2class[w])
                continue

            n_ones = np.where([w.startswith(x) for x in self.range_ones])[0]
            if n_ones.size == 0:
                if "тысяч" in w:
                    t.append(self.num2class["тысяч"])
            else:
                t.append(n_ones[0])

            if any([w.endswith(x) for x in self.ending_teenagers]):
                t.append(self.num2class["teen"])

            elif any([w.endswith(x) for x in self.ending_adults]):
                t.append(self.num2class["adult"])

            elif any([w.endswith(x) for x in self.ending_old]):
                t.append(self.num2class["old"])
        
        t.append(18)  # space at end
        return t
    
    def decode(self, sequence: typing.List[int]) -> int:
        classes_multiplicators = [14, 16, 17]
        classes_sum = [15]
        n = 0
        ans = []
        for class_item in sequence:
            if class_item == 18:
                continue
            if class_item in classes_multiplicators:
                if self.class2num[class_item] > np.sum(ans) > 0:
                        ans.append(n)
                        ans = [np.sum(ans) * self.class2num[class_item]]
                        n = 0
                        continue

                if (n > 0):
                    if self.class2num[class_item] > np.sum(ans):
                        ans.append(n)
                        ans = [np.sum(ans) * self.class2num[class_item]]

                        n = 0
                        continue
                    n *= self.class2num[class_item]
                else:
                    n += self.class2num[class_item]
                    ans.append(n)
                    n = 0
                    continue
                ans.append(n)
                n = 0
                continue

            elif class_item in classes_sum:
                n += 10
                ans.append(n)
                n = 0
                continue

            n += self.class2num[class_item]
        ans.append(n)
        return sum(ans)
    
    def __call__(self, inp: (typing.List[int], int)):
        if isinstance(inp, list):
            return self.decode(inp)
        elif isinstance(inp, int):
            return self.encode(inp)
        else:
            raise NotImplementedError("")
