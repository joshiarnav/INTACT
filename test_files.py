import os
import json

# print(list(os.walk("./MATH")))

def all_problem_files(data_dir):
    """
    Get a list of all problem files in the specified data directory.
    """
    problem_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                problem_files.append(os.path.join(root, file))
    return problem_files

# Write this to a file
print(all_problem_files("./MATH"))

#Write all_problem_files to a text file as a JSON where every list element is a new line
with open('all_problem_files.json', 'w') as filehandle:
    json.dump(all_problem_files("./MATH"), filehandle)
