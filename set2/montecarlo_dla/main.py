import numpy as np
import matplotlib.pyplot as plt

m = 100
n = 100

class RandomWalker:

    def __init__(self, cluster_set):
        self.pos_i = np.random.randint(100)
        self.pos_j = 0
        self.cluster_set = cluster_set
        self.candidate_set = self.init_candidate_set()

    def init_candidate_set(self):
        """Initialize candidate set at walker initialization"""
        candidate_set = set()
        for (i, j) in self.cluster_set:
            neighbors = set()
            neighbors.add((i - 1, j))
            neighbors.add((i + 1, j))
            neighbors.add((i, j - 1))
            neighbor_candidates = neighbors.difference(self.cluster_set)
            candidate_set = candidate_set.union(neighbor_candidates)
        return candidate_set

    def init_pos(self, i, j):
        self.pos_i = i
        self.pos_j = j

    def reset(self):
        self.init_pos(np.random.randint(100), 0)

    def walk(self, nhits):
        assert self.pos_j == 0
        while nhits > 0:
            i, j = self.step()
            if j < 0 or j > m - 1:
                # Abort, set new positions and step again
                self.reset()
            elif (i, j) in self.candidate_set:
                self.cluster_set.add((i, j))
                self.candidate_set.remove((i, j))
                # Add new neighbor cells to candidate set
                neighbors = set()
                if i == 0:
                    neighbors.add((n - 1, j))
                else:
                    neighbors.add((i - 1, j))
                if i == n - 1:
                    neighbors.add((0, j))
                else:
                    neighbors.add((i + 1, j))
                if not j == 0:
                    neighbors.add((i, j - 1))
                if not j == m - 1:
                    neighbors.add((i, j + 1))
                neighbor_candidates = neighbors.difference(self.cluster_set)
                self.candidate_set = self.candidate_set.union(neighbor_candidates)
                print(f"Cluster size: {len(self.cluster_set)}")

                # Init new walker
                self.reset()
                nhits -= 1
            else:
                self.pos_i = i
                self.pos_j = j

    def step(self):
        direction = np.random.randint(4)
        newpos_i = self.pos_i
        newpos_j = self.pos_j

        if direction == 0:
            # Move left
            newpos_i -= 1
            if self.pos_i == 0:
                newpos_i += n
        elif direction == 1:
            # Move right
            newpos_i += 1
            if self.pos_i == n - 1:
                newpos_i = 0
        elif direction == 2:
            # Move up
            newpos_j -= 1
        elif direction == 3:
            # Move down
            newpos_j += 1

        return newpos_i, newpos_j


if __name__ == '__main__':
    # Init candidate set:
    init_cluster_set = {(50, 99)}
    walker = RandomWalker(init_cluster_set)
    walker.walk(700)
    xs = [x[0] for x in walker.cluster_set]
    ys = [x[1] for x in walker.cluster_set]
    plt.xlim(100, 0)
    plt.ylim(100, 0)
    plt.ylabel("M")
    plt.xlabel("N")
    plt.title(f"Cluster size {len(walker.cluster_set)}")
    plt.plot(xs, ys, 'ko', markersize=1)
    plt.show()

