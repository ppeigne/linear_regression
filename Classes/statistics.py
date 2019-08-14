import math


class Statistics:
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(self.dataset)
        self.total = self.sum()
        self.mean = self.total / self.len
        self.median = self.median_measure()
        self.mode = self.mode_measure()
        self.min = min(dataset)
        self.max = max(dataset)
        self.range = self.max - self.min
        self.variance = self.variance_measure()
        self.std = math.sqrt(self.variance)

    def sum(self):
        total = 0
        for x in self.dataset:
            total += x
        return total

    def median_measure(self):
        sorted_set = sorted(self.dataset)
        if self.len % 2 is 0:
            middle = self.len / 2
        else:
            middle = int(self.len / 2 + 0.5)
        return sorted_set[int(middle)]

    def mode_measure(self):
        count = {}
        for x in self.dataset:
            if x not in count:
                count[x] = 1
            else:
                count[x] += 1
        number = 1
        mode = []
        for v, c in count.items():
            if number < c:
                number = c
                mode = [v]
            elif number == c and c > 1:
                mode.append(v)
        if not mode:
            mode = None
        return mode

    def variance_measure(self):
        total = 0
        for x in self.dataset:
            total += (x - self.mean)**2
        return total / (self.len - 1)

    def print_stat(self):
        print("--------------------------------------------")
        print("Data set:\n%s\n" % self.dataset)
        print("Measures of central tendency:")
        print("- Mean: %s" % self.mean)
        print("- Median: %s" % self.median)
        print("- Mode: %s\n" % self.mode)
        #print("Min: %s" % self.min)
        #print("Max: %s" % self.max)
        print("Measures of spread:")
        print("- Range: %s" % self.range)
        print("- Variance: %s" % self.variance)
        print("- Standart deviation: %s" % self.std)
        print("--------------------------------------------")

