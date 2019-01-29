**HOW TO RUN THIS PROGRAM**

1. open terminal
2. switch to python3 environment
3. go to the directory where this ga.py locates
4. make sure the city coordinate file is in the same directory as this ga.py is
5. type in terminal:

> python3 ga.py f_name.txt pop_size muta_rate cross_rate gen

where:

- f_name as the name of the city coordinate file
- pop_size as the population size
- muta_rate as the mutation rate
- cross_rate as the crossover rate
- gen as the max evolution epochs allowed for the program to run


an example of command to run this program would be:

--------------------------------------------------------------
> python3 ga.py A2_TSP_WesternSahara_29.txt 1000 0.05 0.80 1000
--------------------------------------------------------------

**REQUIREMENTS**
Python 3.5 or higher environment with numpy package installed
