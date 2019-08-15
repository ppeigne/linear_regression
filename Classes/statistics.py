import math


class Statistics:
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(self.dataset)
        self.count = self.count_()
        self.total = self.sum()
        self.mean = round(self.total / self.len, 5)

        self.q1, self.median, self.q3 = self.quartiles()
        self.mode = self.mode_measure()
        self.min = self.min_()
        self.max = self.max_()
        self.range = self.max - self.min
        self.variance = round(self.variance_measure(), 5)
        self.std = round(math.sqrt(self.variance), 5)

    def min_(self):
        # m = self.dataset[0]
        # for x in range(1, self.len):
        #     if self.dataset[x] < m:
        #         m = x

        return sorted(self.dataset)[0]

    def max_(self):
        # m = self.dataset[0]
        # for x in range(1, self.len):
        #     if self.dataset[x] > m:
        #         m = x
        return sorted(self.dataset)[self.len - 1]
        #return m

    def count_(self):
        n = 1
        while n in range(0, self.len):
            if self.dataset[n] is not None:
                n += 1
        return n

    def sum(self):
        total = 0
        for x in self.dataset:
            total += x
        return total

    def quartiles(self):
        sorted_set = sorted(self.dataset)
        if self.len % 4 == 0:
            q1 = int(self.len / 4)
        else:
            q1 = int(self.len / 4) + 1

        if self.len % 2 == 0:
            q2 = int(self.len / 2)
        else:
            q2 = int(self.len / 2) + 1

        if self.len % 4 == 0:
            q3 = int(3 * self.len / 4)
        else:
            q3 = int(3 * self.len / 4) + 1

        return sorted_set[q1], sorted_set[q2], sorted_set[q3]

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
        print("Spread:")
        print("- Range: %s" % self.range)
        print("- Var: %s" % self.variance)
        print("- Std: %s\n" % self.std)

     #   print("Data set:\n%s\n" % sorted(self.dataset))
        print("Central tendency:")
        print("- Mean: %s" % self.mean)
        print("- Med: %s" % self.median)
        print("- Mode: %s\n" % self.mode)

        print("Distribution:")
        print("- Min: %s" % self.min)
        print("- 25%%: %s" % self.q1)
        print("- 50%%: %s" % self.median)
        print("- 75%%: %s" % self.q3)
        print("- Max: %s" % self.max)

        print("--------------------------------------------")


