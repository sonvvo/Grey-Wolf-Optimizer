############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Grey Wolf Optimizer

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Grey_Wolf_Optimizer, File: Python-MH-Grey Wolf Optimizer.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Grey_Wolf_Optimizer>

############################################################################

# Required Libraries
import numpy as np
import math
import random
import os


# Function
def target_function():
    return


# Function: Initialize Variables
def initial_position(
    pack_size=5, min_values=[-5, -5], max_values=[5, 5], target_function=target_function
):
    position = np.zeros((pack_size, len(min_values) + 1))
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0 : position.shape[1] - 1])
    return position


# Function: Initialize Alpha
def alpha_position(dimension=2, target_function=target_function):
    alpha = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        alpha[0, j] = 0.0
    alpha[0, -1] = target_function(alpha[0, 0 : alpha.shape[1] - 1])
    return alpha


# Function: Initialize Beta
def beta_position(dimension=2, target_function=target_function):
    beta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        beta[0, j] = 0.0
    beta[0, -1] = target_function(beta[0, 0 : beta.shape[1] - 1])
    return beta


# Function: Initialize Delta
def delta_position(dimension=2, target_function=target_function):
    delta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        delta[0, j] = 0.0
    delta[0, -1] = target_function(delta[0, 0 : delta.shape[1] - 1])
    return delta


# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if updated_position[i, -1] < alpha[0, -1]:
            alpha[0, :] = np.copy(updated_position[i, :])
        if (
            updated_position[i, -1] > alpha[0, -1]
            and updated_position[i, -1] < beta[0, -1]
        ):
            beta[0, :] = np.copy(updated_position[i, :])
        if (
            updated_position[i, -1] > alpha[0, -1]
            and updated_position[i, -1] > beta[0, -1]
            and updated_position[i, -1] < delta[0, -1]
        ):
            delta[0, :] = np.copy(updated_position[i, :])
    return alpha, beta, delta


# Function: Updtade Position
def update_position(
    position,
    alpha,
    beta,
    delta,
    a_linear_component=2,
    min_values=[-5, -5],
    max_values=[5, 5],
    target_function=target_function,
):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range(0, len(min_values)):
            r1_alpha = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            r2_alpha = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            a_alpha = 2 * a_linear_component * r1_alpha - a_linear_component
            c_alpha = 2 * r2_alpha
            distance_alpha = abs(c_alpha * alpha[0, j] - position[i, j])
            x1 = alpha[0, j] - a_alpha * distance_alpha
            r1_beta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            r2_beta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            a_beta = 2 * a_linear_component * r1_beta - a_linear_component
            c_beta = 2 * r2_beta
            distance_beta = abs(c_beta * beta[0, j] - position[i, j])
            x2 = beta[0, j] - a_beta * distance_beta
            r1_delta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            r2_delta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            a_delta = 2 * a_linear_component * r1_delta - a_linear_component
            c_delta = 2 * r2_delta
            distance_delta = abs(c_delta * delta[0, j] - position[i, j])
            x3 = delta[0, j] - a_delta * distance_delta
            updated_position[i, j] = np.clip(
                ((x1 + x2 + x3) / 3), min_values[j], max_values[j]
            )
        updated_position[i, -1] = target_function(
            updated_position[i, 0 : updated_position.shape[1] - 1]
        )
    return updated_position


# GWO Function
def grey_wolf_optimizer(
    pack_size=5,
    min_values=[-5, -5],
    max_values=[5, 5],
    iterations=50,
    target_function=target_function,
):
    count = 0
    alpha = alpha_position(dimension=len(min_values), target_function=target_function)
    beta = beta_position(dimension=len(min_values), target_function=target_function)
    delta = delta_position(dimension=len(min_values), target_function=target_function)
    position = initial_position(
        pack_size=pack_size,
        min_values=min_values,
        max_values=max_values,
        target_function=target_function,
    )
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", alpha[-1])
        a_linear_component = 2 - count * (2 / iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position = update_position(
            position,
            alpha,
            beta,
            delta,
            a_linear_component=a_linear_component,
            min_values=min_values,
            max_values=max_values,
            target_function=target_function,
        )
        count = count + 1
    print(alpha[-1])
    return alpha


############################################################################

# job scheduling problem parameters
num_employees = 6
num_of_working_days = 5  # 5 days in a week
num_variables = num_employees * num_of_working_days


# fitness func for job scheduling
def scheduling_fitness(schedule):
    schedule = schedule.reshape(num_employees, num_of_working_days)
    cost = 0

    for emp_sched in schedule.T:
        # at least one employee work on a week day, otherwise
        # company will lose RM1000
        if np.sum(np.round(emp_sched)) < 1:
            cost += 1000  # penalty for uncovered shift

        # no more than two employees work on the same day
        if np.sum(np.round(emp_sched)) > 2:
            cost += 1000 * (
                np.sum(np.round(emp_sched)) - 2
            )  # company will lose RM1000 to pay for each extra employee
            # work on that day (start counting from the third employee)

    # no consecutive work days for any employee
    for emp_sched in schedule:
        for i in range(len(emp_sched) - 1):
            if round(emp_sched[i]) == 1 and round(emp_sched[i + 1]) == 1:
                # company lose RM1000 for any employee works two days in a row
                cost += 1000

    # each employee must work at least one day
    for emp_sched in schedule:
        if np.sum(np.round(emp_sched)) == 0:
            # company lose RM1000 if an employee does not work at all in that
            # week due to some management fee
            cost += 1000

    return cost


# GWO parameters
num_iterations = 50
population_size = 30

# Optimize using GWO
best_solution = grey_wolf_optimizer(
    pack_size=population_size,
    min_values=[0 for _ in range(num_variables)],  # 0 for not working
    max_values=[1 for _ in range(num_variables)],  # 1 for working
    iterations=num_iterations,
    target_function=scheduling_fitness,
)[
    0, :-1
]  # get only the positions, not the fitness

# convert solution back to scheduling format
best_schedule = np.round(best_solution).reshape(num_employees, num_of_working_days)


# Display results
print(f"\nBest Fitness (Cost Penalty): RM{scheduling_fitness(best_solution)}")
print(f"Best Working Schedule:\n{best_schedule}\n")


# func to display the job scheduling
def display_schedule(schedule):
    print(f"Company GWO with {num_employees} employee(s)")
    print("This week working schedule for each employee:")
    working_days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for i, employee in enumerate(schedule):
        print(f"Employee {i+1}:", end=" ")
        for day, work in zip(working_days, employee):
            if work:
                print(day, end=" ")
        print()


display_schedule(best_schedule)
print()
