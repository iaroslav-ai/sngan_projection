import numpy as np

from skopt.space import Real, Categorical, Integer, Space
from skopt.utils import create_result

class RandomStepOptimizer():
    """
    A method that arised from the empirical evaluation of the 
    performance of GeneticOptimizer class.
    It appears that on the set of benchmarks that are used in
    this library, the mutation operation when used alone, that
    is, without the crossover step, yields the best performing
    results. This surprising observation was obtained by running 
    optimization of ga_optimize parameters using Bayesian 
    Optimization. 
    For  more details, refer to find_hypers_ga.py file in this folder.
    The result was that Bayesian Optimization algorithm 
    consistently selected the tournament size to being 1.0,
    which is equivalent to always work on top performing vector,
    and disabling the crossover procedure. The optimization
    of the GA hyperparameters was done on ~30 tasks of the 
    evalset set of benchmarks, and then was validated on the
    rest of ~20 tasks. The ranking of the algorithm was
    fairly close on both partitions of functions:
    - "training" partition: ranking of 2.1 (less is better)
    - testing partition: ranking of ~2.5
    - max ranking was obtained with dummy optimizer of ~5.5
    Parameters
    ----------
    * `n_random_starts` [int]:
        Size of initial population, generated at random. After
        this number of fitness values is reported for individuals
        (points), the actual algorithm starts.
    * `mutation_rate` [float]:
        Probability of a value of dimension of a point (gene of 
        individual) to be randomly changed. 
        
    """
    def __init__(self, dimensions, n_random_starts=10, 
        mutation_rate=0.05):
        self.Xi = []
        self.yi = []
        self.space = Space(dimensions)
        self.n_random_starts = n_random_starts
        self.n_random_starts_ = n_random_starts
        self.mutation_rate = mutation_rate
    
    def rnd(self, prob=None):
        """Returns true with probability given by 'prob' """
        if prob is None:
            prob = self.mutation_rate

        return np.random.rand() < prob

    def mutate(self, point):
        """Perform random mutation on the point"""

        finished = False
        while not finished:
            # there is a large chance that the point will not be mutated.
            # but it does not really makes sense to evaluate the same point
            # multiple times (at least for noiseless case) hence the point
            # should necessary be mutated.
            result = []
            for i in range(len(point)):
                dim = self.space.dimensions[i]
                v = point[i]

                if isinstance(dim, Categorical) and self.rnd():
                    # record that there has been a change to the point
                    finished = True 
                    i = np.random.randint(len(dim.categories))
                    v = dim.categories[i]
                
                if isinstance(dim, Real) or isinstance(dim, Integer):
                    # one could represent real value as a sequence
                    # of binary values. however, here mutation is done
                    # explicity with numbers.
                    a, b = dim.low, dim.high
                    for i in range(-16, 1):
                        if self.rnd():
                            # record that there has been a change to the point
                            finished = True                     
                            # amount of change, proportional to the size of the dimension
                            scale = 2.0 ** i

                            # determine the direction of change - towards a or b
                            if self.rnd(0.5):
                                v += (b-v) * scale
                            else:
                                v -= (v-a) * scale
                            
                            # it should always hold that a <= v <= b.    
                            if v < a:
                                v = a
                            
                            if v > b:
                                v = b
                    
                    # round the number if integer dimension
                    if isinstance(dim, Integer):
                        v = round(v)
                
                result.append(v)

        return result

    def ask(self):
        # a standard random initialization
        if self.n_random_starts_ > 0:
            sample = self.space.rvs()[0]
            return sample
        
        # find the best point
        c = min(zip(self.Xi, self.yi), key=lambda t: t[-1])[0]

        # mutate the point
        c = self.mutate(c)
        return c
    
    def tell(self, x, y):
        self.n_random_starts_ = max(0, self.n_random_starts_ - 1)
        self.Xi.append(x)
        self.yi.append(y)

def rs_minimize(func, dimensions, n_calls=64, n_random_starts=13, mutation_rate=0.1):
    """
    Use the RandomStepOptimizer here
    """
    optimizer = RandomStepOptimizer(
        dimensions=dimensions, 
        n_random_starts=n_random_starts,
        mutation_rate=mutation_rate
    )

    for i in range(n_calls):
        x = optimizer.ask()
        y = func(x)
        optimizer.tell(x, y)

    return create_result(optimizer.Xi, optimizer.yi, dimensions)
