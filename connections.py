from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
from itertools import combinations


i = 1
filename = f'game_data_{i}.json'
filepath = os.path.join('data', filename)

with open(filepath, 'r') as file:
    data_json = json.load(file)

# Accessing the game data
game_data = data_json['props']['pageProps']['answers']

words = np.array([x['words'] for x in game_data]).ravel()
np.random.shuffle(words)
print(words)

# all_combs = list(combinations(words, 4))

# Convert words to embeddings using a pre-trained model
model = SentenceTransformer('all-mpnet-base-v2')

# Define a function to form a group of 4 words
def form_group(words, model):
    # Encode words
    embeddings = model.encode(words)
    sim = cosine_similarity(embeddings)
    
    # Ignore diagonal (self-similarity) by setting it to -1
    np.fill_diagonal(sim, -1)
    
    # Find the initial pair with the highest similarity
    max_sim_indices = np.unravel_index(np.argmax(sim), sim.shape)
    group = [words[max_sim_indices[0]], words[max_sim_indices[1]]]
    
    # Remove the initial pair from the pool
    remaining_words = np.delete(words, [max_sim_indices[0], max_sim_indices[1]])
    
    # Iteratively add words to the group based on highest similarity to the group
    while len(group) < 4:
        # Recalculate similarity between the group and remaining words
        group_embeddings = model.encode(group)
        remaining_embeddings = model.encode(remaining_words)
        sim_to_group = cosine_similarity(remaining_embeddings, group_embeddings)
        avg_sim_to_group = np.mean(sim_to_group, axis=1)  # Average similarity to the group
        
        # Select the word with the highest average similarity to the group
        next_word_index = np.argmax(avg_sim_to_group)
        group.append(remaining_words[next_word_index])
        
        # Remove selected word from the pool of remaining words
        remaining_words = np.delete(remaining_words, next_word_index)
    
    return group, remaining_words


def check_correctness(groups, game_data):
    correct_count = 0
    for group in groups:
        # For each group, check if there's a matching description in the game_data
        for entry in game_data:
            if set(group).issubset(set(entry['words'])):
                correct_count += 1
                break  # Found a match, move to the next group
    return correct_count

# Form the four groups in a loop
groups = []
remaining_words = words.copy()  # Ensure we have a mutable list that starts with all words

for _ in range(4):
    group, remaining_words = form_group(remaining_words, model)
    groups.append(group)

# Print the formed groups
for i, group in enumerate(groups, start=1):
    print(f"Group {i} formed: {group}")

# Check correctness
correct_count = check_correctness(groups, game_data)
print(f"\nNumber of correct groups: {correct_count} out of 4")

# Print the solution for reference
print('\n--- Solution ---')
for x in game_data:
    print(x['description'], x['words'])




def calculate_coherence(group, embeddings, model):
    group_embeddings = model.encode([model for model in group])
    sim_matrix = cosine_similarity(group_embeddings)
    np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
    average_similarity = np.mean(sim_matrix)
    return average_similarity

def cross_check_groups(groups, model):
    group_embeddings = [model.encode([" ".join(group)]) for group in groups]  # One embedding per group
    improvements = True

    while improvements:
        improvements = False
        for i, group_i in enumerate(groups[:-1]):
            for j, group_j in enumerate(groups[i+1:], start=i+1):
                # Calculate current coherence
                current_coherence_i = calculate_coherence(group_i, group_embeddings[i], model)
                current_coherence_j = calculate_coherence(group_j, group_embeddings[j], model)

                # Try swapping each element and check if it improves coherence
                for element_i in group_i:
                    for element_j in group_j:
                        new_group_i = [element_j if x==element_i else x for x in group_i]
                        new_group_j = [element_i if x==element_j else x for x in group_j]
                        new_coherence_i = calculate_coherence(new_group_i, group_embeddings[i], model)
                        new_coherence_j = calculate_coherence(new_group_j, group_embeddings[j], model)

                        # Check if swapping improves overall coherence
                        if new_coherence_i + new_coherence_j > current_coherence_i + current_coherence_j:
                            groups[i], groups[j] = new_group_i, new_group_j
                            improvements = True
                            break  # Break out of the innermost loop
                    if improvements:
                        break  # Break out of the next loop if an improvement was made

    return groups

# Assuming groups is a list of your formed groups and model is your SentenceTransformer model
optimized_groups = cross_check_groups(groups, model)
print(optimized_groups)  # Print the optimized groups