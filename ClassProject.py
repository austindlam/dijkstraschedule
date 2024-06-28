import itertools
import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Campus 5x5 grid
grid_size = 5

# Positions of all available classrooms on campus
classrooms = [(0, 0), (1, 3), (4, 4)]

# 3 PM classes
classes_3pm = ['A', 'B', 'C']
print("Classes at 3pm:", classes_3pm)

# 4 PM classes
classes_4pm = ['D', 'E', 'F']
print("Classes at 4pm:", classes_4pm)

# 5 PM classes
classes_5pm = ['G', 'H', 'I']
print("Classes at 5pm:", classes_5pm)

# Generate permutations for available classrooms
def generate_classroom_placements(classes):
    return list(itertools.permutations(classrooms, len(classes)))

# 3PM permutations
perms_3pm = generate_classroom_placements(classes_3pm)

# 4PM permutations
perms_4pm = generate_classroom_placements(classes_4pm)

# 5PM permutations
perms_5pm = generate_classroom_placements(classes_5pm)

# Generate student schedules
def generate_student_schedules(num_schedules=10):

    # List of all class groups for 3PM, 4PM, and 5PM
    all_class_groups = [classes_3pm, classes_4pm, classes_5pm]

    # List to store generated schedules
    schedules = []

    # Loop to generate the specified number of schedules
    for student_id in range(1, num_schedules + 1):

        # Randomly choose whether the student will have 2 or 3 classes
        num_classes = random.choice([2, 3])

        # Initialize an empty schedule for the student
        schedule = []

        # Randomly pick time slots for the number of classes
        time_slots = random.sample(all_class_groups, num_classes)

        # Loop through each time slot and randomly pick a lecture from it
        for lecture in time_slots:
            schedule.append(random.choice(lecture))

        # Sort the schedule and add the student ID
        schedules.append((student_id, sorted(schedule)))

    return schedules

# Generate student schedules
student_schedules = generate_student_schedules()

# Print the generated student schedules
print("Generated Student Schedules:")
for student_id, schedule in student_schedules:
    print(f"Student {student_id}: {schedule}")

# Find the shortest path using Dijkstra's algorithm
def dijkstra(start, end):

    # Priority queue to store nodes
    pq = [(0, start)]

    # Dictionary to store shortest distance to each node
    distances = {start: 0}

    # Loop until the priority queue is empty
    while pq:

        # Pop node with smallest distance
        current_distance, current_node = heapq.heappop(pq)

        # If the current node is the end node, return
        if current_node == end:
            return current_distance

        # Iterate through neighbors of current node
        for neighbor in get_neighbors(current_node, grid_size):

            # Calculate distance to the neighbor
            distance = current_distance + 1

            # If the neighbor has not been visited or a shorter path is found
            if neighbor not in distances or distance < distances[neighbor]:

                # Update the shortest distance to the neighbor
                distances[neighbor] = distance

                # Push neighbor and its distance to the priority queue
                heapq.heappush(pq, (distance, neighbor))

    return float('inf')

# Get neighbors of a node
def get_neighbors(node, grid_size):
    neighbors = []
    x, y = node # Current position
    if x > 0:
        neighbors.append((x-1, y)) # Above
    if x < grid_size - 1:
        neighbors.append((x+1, y)) # Below
    if y > 0:
        neighbors.append((x, y-1)) # Left
    if y < grid_size - 1:
        neighbors.append((x, y+1)) # Right
    return neighbors

# Calculate distance for each grid layout
def calculate_distances(placements, schedules):

    total_distance = 0

    # Distances list
    distances = []

    # Loop through each student schedule (omit the studentID)
    for _, schedule in schedules:

        schedule_distance = 0

        # Loop through the schedule to calculate the distance between back to back classes
        for i in range(len(schedule) - 1):

            # Positions of back to back classes
            pos1 = placements[schedule[i]]
            pos2 = placements[schedule[i + 1]]

            # Use dijkstra's to find shortest path
            distance = dijkstra(pos1, pos2)

            # Add distance to schedule distance
            schedule_distance += distance

        # Append schedule distance to distances list
        distances.append(schedule_distance)

        # Add schedule distance to the total distance
        total_distance += schedule_distance

    # Calculate average distance
    average_distance = total_distance / len(schedules)

    return average_distance, distances

# Convert placements to dict for easier access
def convert_placements_to_dict(placements, classes):
    return {cls: placements[i] for i, cls in enumerate(classes)}

# Evaluate all grids and find the best grid
def evaluate_grids():

    best_grid = None
    lowest_average_distance = float('inf')
    highest_average_distance = float(0)
    all_avg_distances = []
    all_grids = []
    all_schedule_distances = []

    # Iterate through all possible placements at 3 PM, 4 PM, and 5 PM
    for i in perms_3pm:
        for j in perms_4pm:
            for k in perms_5pm:
                # Combine placements into a single dictionary
                all_placements = convert_placements_to_dict(i, classes_3pm)
                all_placements.update(convert_placements_to_dict(j, classes_4pm))
                all_placements.update(convert_placements_to_dict(k, classes_5pm))

                # Calculate average distance and schedule distances for current grid layout
                avg_distance, schedule_distances = calculate_distances(all_placements, student_schedules)

                # Store average distance, grid layout, and schedule distances
                all_avg_distances.append(avg_distance)
                all_grids.append(all_placements)
                all_schedule_distances.append(schedule_distances)

                # Set best grid if a lower average distance is found
                if avg_distance < lowest_average_distance:
                    lowest_average_distance = avg_distance
                    best_grid = all_placements

                # Set highest average distance if a higher average distance is found
                if avg_distance > highest_average_distance:
                    highest_average_distance = avg_distance

    return best_grid, lowest_average_distance, all_avg_distances, all_grids, highest_average_distance, all_schedule_distances

# Evaluate grids
best_grid, lowest_avg_distance, all_avg_distances, all_grids, highest_avg_distance, all_schedule_distances = evaluate_grids()

# Calculate average of all average distances
average_avg_distance = sum(all_avg_distances) / len(all_avg_distances)

# Print the results
print("Best Grid Configuration:", best_grid)
print("Lowest Average Distance:", lowest_avg_distance)
print("Average Average Distance:", average_avg_distance)
print("Highest Average Distance:", highest_avg_distance)
print("All Average Distances:", all_avg_distances)

# Plot average distances
plt.figure(figsize=(12, 6))
plt.plot(all_avg_distances, marker='o')
plt.title('Average Distance for Each Grid')
plt.xlabel('Grid Index')
plt.ylabel('Average Distance')
plt.grid(True)
plt.show()

# Plot grid with class positions
def plot_grid(grid, classes, title):

    # Initialize empty grid
    grid_plot = np.full((grid_size, grid_size), '', dtype=object)

    # Place the classes in the grid
    for cls in classes:
        x, y = grid[cls]
        grid_plot[x, y] = cls

    # Create grid
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(np.zeros((grid_size, grid_size)), cmap='Greys')

    # Place class labels
    for (i, j), val in np.ndenumerate(grid_plot):
        if val:
            ax.text(j, i, val, ha='center', va='center', fontsize=12, color='red')
        else:
            ax.text(j, i, '', ha='center', va='center', fontsize=12, color='black')

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    plt.title(title)
    plt.show()

# Plot grid with available classrooms
def plot_grid_with_Xs(positions, grid_size, title):

    # Initialize empty grid
    grid_plot = np.full((grid_size, grid_size), '', dtype=object)

    # Place X at specified positions
    for pos in positions:
        x, y = pos
        grid_plot[x, y] = 'X'

    # Create grid
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(np.zeros((grid_size, grid_size)), cmap='Greys')

    # Place class labels
    for (i, j), val in np.ndenumerate(grid_plot):
        if val:
            ax.text(j, i, val, ha='center', va='center', fontsize=12, color='red')

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    plt.title(title)
    plt.show()

# Plot grids
plot_grid(best_grid, classes_3pm, "Best Grid Configuration at 3pm")
plot_grid(best_grid, classes_4pm, "Best Grid Configuration at 4pm")
plot_grid(best_grid, classes_5pm, "Best Grid Configuration at 5pm")

# Plot available classroom grid
plot_grid_with_Xs(classrooms, grid_size, "Available Classrooms")

# Create table
data = []
columns = ["Grid Index"] + [f"Student {i+1}" for i in range(len(student_schedules))] + ["Average Distance"]

# Populate table
for idx, (avg_distance, distances) in enumerate(zip(all_avg_distances, all_schedule_distances)):
    row = [idx] + distances + [avg_distance]
    data.append(row)

# Create DataFrame from the data
df = pd.DataFrame(data, columns=columns)

# Create 5 number summary
summary_data = []
for idx, distances in enumerate(all_schedule_distances):
    min_val = min(distances)
    q1 = np.percentile(distances, 25)
    median_val = np.median(distances)
    q3 = np.percentile(distances, 75)
    max_val = max(distances)
    summary_data.append([min_val, q1, median_val, q3, max_val])

# Summary columns
summary_columns = ["Min", "Q1", "Median", "Q3", "Max"]

# Create DataFrame for the summary data
summary_df = pd.DataFrame(summary_data, columns=summary_columns)

# Concatenate data
df_summary = pd.concat([df, summary_df], axis=1)

# Highlight min/max
def highlight_avg_min_max(s):
    is_min = s == s.min()
    is_max = s == s.max()
    return ['background-color: yellow' if v else 'background-color: lightgreen' if m else '' for v, m in zip(is_max, is_min)]

# Highlight min/max average distance
df_style = df_summary.style.apply(highlight_avg_min_max, subset=["Average Distance"])

df_style