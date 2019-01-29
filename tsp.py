import random
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name, visu_city = 0):
    '''read city coordinates from a txt file
    txt file should be in the same folder where this program locates
    data format should be 'city_index x_coordinate y_coordinate'
    the output are three np.arraies which represent
    city_index x_coordinate y_coordinate accordingly
    '''
    if not isinstance(file_name, str):
        raise TypeError('bad operand type')
    else:
        city_ind = []
        x_cord = []
        y_cord = []
        with open(file_name,'r') as f:
            for line in f:
                row = line.split()
                city_ind.append(int(row[0]))
                x_cord.append(float(row[1]))
                y_cord.append(float(row[2]))
        n_city = len(city_ind)
        x_cord = np.asarray(x_cord)
        y_cord = np.asarray(y_cord)
        if visu_city == 1:
            plt.scatter(x_cord, y_cord, c='r')
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.title('Layout of cities')
            plt.show()
        return x_cord, y_cord, n_city


def init_pop(pop_size, n_city):
    '''initialize the 1st generation of population
    the output is a pop_size * number_of_cities matrix
    where each row represents a possible visit sequence
    pop_size should be an positive integer number'''
    if not (isinstance(pop_size, int) and pop_size>=1):
        raise TypeError('bad operand type')
    else:
        pop = np.zeros((pop_size, n_city), dtype = int)
        for i in range(pop_size):
            l = np.arange(1, n_city+1, dtype = int)
            np.random.shuffle(l)
            pop[i] = l
    return pop


def leng(order, x_cord, y_cord):
    '''given a visit order of cities, compute the total travel length'''
    l = 0
    for i in range(len(order)-1):
        l += np.sqrt(np.square(x_cord[order[i]-1] - x_cord[order[i+1]-1]) \
        + np.square(y_cord[order[i]-1] - y_cord[order[i+1]-1]))
    l += np.sqrt(np.square(x_cord[order[-1]-1] - x_cord[order[0]-1]) + \
    np.square(y_cord[order[-1]-1] - y_cord[order[0]-1]))
    return l


def fitness(pop, x_cord, y_cord):
    '''given the population, compute their fitness
    the output is a column vector where each row represents
    the fitness value of that visit order
    fitness_value = exp(avg_length/l * 5)
    higher fitness value means fitter indivilual'''
    l = np.zeros((pop.shape[0]))
    for i in range(pop.shape[0]):
        l[i] = leng(pop[i], x_cord, y_cord)
    min_length = np.amin(l, axis = 0)
    max_length = np.amax(l, axis = 0)
    avg_length = np.average(l, axis = 0)
    fit = np.exp(avg_length/l * 5)
    return fit, [min_length, avg_length, max_length]


def sel_par(pop, fit):
    '''select mating parent from the whole population
    the probability of being chosen is based on their fitness
    indivilual with higher fitness value is more likely to be chosen
    prob_indiv = fit_indiv/sum(fit)
    output will be the visit sequence of chosen parent
    '''
    # compute the probability of each indivilual based on their fitness
    prob = fit/np.sum(fit)
    # generate a random number between 0 and 1
    rand = random.uniform(0,1)
    start = 0
    for i in range(prob.shape[0]):
        end = start + prob[i]
        if (rand >= start) and (rand <= end):
            par = pop[i]
            return par
        start = end


def cross(par1, par2):
    '''use order crossover to generate offspring'''
    # randomly pick two crossover points
    rand_1 = random.randint(0, par1.shape[0]-1)
    rand_2 = random.randint(0, par1.shape[0]-1)
    if rand_1 <= rand_2:
        start = rand_1
        end = rand_2
    else:
        start = rand_2
        end = rand_1
    off = np.zeros(par1.shape[0], int)
    # copy the selected segment from par1
    for i in range(start, end+1):
        off[i] = par1[i]
    # create fill list from par2
    fill = np.zeros(par1.shape[0])
    ind = end
    for i in range(0, par1.shape[0]):
        if ind == par1.shape[0]-1:
            ind = 0
        else:
            ind += 1
        fill[i] = par2[ind]
    # fill in ohter alleles from par2
    ind2 = end
    for i in range(fill.shape[0]):
        if not (fill[i] in off):
            if ind2 == par1.shape[0]-1:
                ind2 = 0
            else:
                ind2 += 1
            off[ind2] = fill[i]
    return off


def pmx(par1, par2):
    # randomly pick two crossover points
    rand_1 = random.randint(0, par1.shape[0]-1)
    rand_2 = random.randint(0, par1.shape[0]-1)
    if rand_1 <= rand_2:
        start = rand_1
        end = rand_2
    else:
        start = rand_2
        end = rand_1
    off1 = np.zeros(par1.shape[0], int)
    # copy the selected segment from par1
    for i in range(start, end+1):
        off1[i] = par1[i]
    # fill the elements in the selected segment from par2 into off1
    for i in range(start, end+1):
        if not (par2[i] in off1):
            # find the index to be filled into
            ind = int(np.argwhere(par2 == off1[i]))
            while off1[ind] != 0:
                ind = int(np.argwhere(par2 == off1[ind]))
            off1[ind] = par2[i]
    # fill in other elements outside the segment from par2
    for i in range(off1.shape[0]):
        if off1[i] == 0:
            off1[i] = par2[i]
    return off1


def muta(off1, muta_rate):
    '''
    swap mutation for offspring
    each allele has a mutation rate of muta_rate
    when an allele is chosen, swap the value with
    another random allele in the sequence
    '''
    mut_off = np.copy(off1)
    for i in range(off1.shape[0]):
        rand = random.uniform(0,1)
        if rand < muta_rate:
            s = random.randint(0, off1.shape[0]-1)
            tmp = mut_off[i]
            mut_off[i] = mut_off[s]
            mut_off[s] = tmp
    return mut_off


def muta_new(off1, muta_rate):
    '''
    swap mutation for offspring
    each allele has a mutation rate of muta_rate
    when an allele is chosen, swap the value with
    another random allele in the sequence
    '''
    mut_off = np.copy(off1)
    rand = random.uniform(0,1)
    if rand <= muta_rate:
        ind1 = random.randint(0, off1.shape[0]-1)
        ind2 = random.randint(0, off1.shape[0]-1)
        tmp = mut_off[ind1]
        mut_off[ind1] = mut_off[ind2]
        mut_off[ind2] = tmp
    return mut_off


def reprod(pop, fit, cross_rate, muta_rate):
    '''
    apply roulette selection to select parents for the mating pool
    generate offsprings by applying order crossover and swap mutation
    the number of offsprings is equal to parents
    '''
    # create mating pool (pool size = pop size)
    pop_size = pop.shape[0]
    if pop_size % 2 != 0:
        mate_size = pop_size + 1
    else:
        mate_size = pop_size
    pool = np.zeros([mate_size,pop.shape[1]], int)
    for i in range(mate_size):
        pool[i] = sel_par(pop, fit)
    np.random.shuffle(pool)
    # create offsprings
    off = np.zeros(pool.shape, int)
    # apply order crossover to each consecutive
    # pair with probability cross_rate
    for i in range(0, mate_size-1, 2):
        rand = random.uniform(0,1)
        if rand <= cross_rate:
            off[i] = pmx(pool[i], pool[i+1])
            off[i+1] = pmx(pool[i+1], pool[i])
        else:
            off[i] = pool[i]
            off[i+1] = pool[i+1]
        # apply swap mutation to each offspring
        # with probability muta_rate
        off[i] = muta_new(off[i], muta_rate)
        off[i+1] = muta_new(off[i+1], muta_rate)
    return off


def nchoose_pop(pop, off, x_cord, y_cord):
    '''
    choose indiviluals for next generation
    combine the parents and generated offsprings
    ranking the whole population and choose
    the top pop_size fitter indiviluals
    '''
    # concatenate parents and offsprings
    con = np.concatenate((pop,off), axis=0)
    # compute the fitness
    fit_con, _ = fitness(con, x_cord, y_cord)
    # get the indices of ranked fitness (in asending order)
    rank_index = np.argsort(fit_con)
    new_pop = np.zeros(pop.shape, int)
    n = new_pop.shape[0]
    tn = int(n * 3 / 5)
    rn = n - tn
    # choose top 3/5 * n (pop_size) fitter indiviluals
    for i in range(tn):
        new_pop[i, :] = con[rank_index[-i-1],:]
    # randomly choose 2/5 * n from the rest set
    for j in range(rn):
        ind = random.randint(0, rn+1)
        new_pop[tn+j] = con[ind]
    return new_pop
