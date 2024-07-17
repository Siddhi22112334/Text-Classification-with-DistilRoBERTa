import os
import numpy as np
import dill as pickle

models = ["gpt", "claude"]
domains = ["wp", "reuter", "essay"]

def get_file_list(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_list.append(os.path.join(root, file))
    return file_list

def prepare_data(seed=0):
    np.random.seed(seed)

    datasets = {
        "wp": ["data/wp/human", "data/wp/gpt"],
        "reuter": ["data/reuter/human", "data/reuter/gpt"],
        "essay": ["data/essay/human", "data/essay/gpt"]
    }

    files = []
    labels = []

    for domain, paths in datasets.items():
        for path in paths:
            label = 1 if "gpt" in path else 0
            domain_files = get_file_list(path)
            files.extend(domain_files)
            labels.extend([label] * len(domain_files))

    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    train_indices = indices[: int(0.8 * len(indices))]
    test_indices = indices[int(0.8 * len(indices)):]

    indices_dict = {}

    for model in models + ["human"]:
        train_idx, test_idx = get_indices(files, train_indices, test_indices, model)
        indices_dict[f"{model}_train"] = train_idx
        indices_dict[f"{model}_test"] = test_idx

    for model in models + ["human"]:
        for domain in domains:
            train_key = f"{model}_{domain}_train"
            test_key = f"{model}_{domain}_test"
            train_idx, test_idx = get_indices(files, train_indices, test_indices, model, domain)
            indices_dict[train_key] = train_idx
            indices_dict[test_key] = test_idx

    os.makedirs('data', exist_ok=True)

    with open('data/indices_dict.pkl', 'wb') as f:
        pickle.dump(indices_dict, f)

    with open('data/files.pkl', 'wb') as f:
        pickle.dump(files, f)

    with open('data/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    

def get_indices(files, train_indices, test_indices, model, domain=None):
    if domain:
        filter_fn = lambda file: domain in file and model in file
    else:
        filter_fn = lambda file: model in file
    where = [i for i, file in enumerate(files) if filter_fn(file)]
    curr_train = [i for i in train_indices if i in where]
    curr_test = [i for i in test_indices if i in where]
    return curr_train, curr_test

if __name__ == "__main__":
    prepare_data()