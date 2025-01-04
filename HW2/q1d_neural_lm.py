import os
import random
import time

import numpy as np
import pandas as pd

from data_utils import utils
from sgd import sgd
from q1c_neural import forward, forward_backward_prop

# Set the working directory to the path where your script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
VOCAB_EMBEDDING_PATH = "data/lm/vocab.embeddings.glove.txt"
BATCH_SIZE = 50
NUM_OF_SGD_ITERATIONS = 40000
LEARNING_RATE = 0.3


def load_vocab_embeddings(path=VOCAB_EMBEDDING_PATH):
    result = []
    with open(path) as f:
        index = 0
        for line in f:
            line = line.strip()
            row = line.split()
            data = [float(x) for x in row[1:]]
            assert len(data) == 50
            result.append(data)
            index += 1
    return result


def load_data_as_sentences(path, word_to_num):
    """
    Conv:erts the training data to an array of integer arrays.
      args:
        path: string pointing to the training data
        word_to_num: A dictionary from string words to integers
      returns:
        An array of integer arrays. Each array is a sentence and each
        integer is a word.
    """
    docs_data = utils.load_dataset(path)
    S_data = utils.docs_to_indices(docs_data, word_to_num)
    return docs_data, S_data

def load_data_as_sentences_q3c(path, word_to_num):
    """
    This function is used in q3c only!
    Converts the training data to an array of integer arrays.
      args:
        path: string pointing to the training data - in this case shakespeare / wikipedia _for_perplexity
        word_to_num: A dictionary from string words to integers
      returns:
        An array of integer arrays. Each array is a sentence and each
        integer is a word.
    """
    docs_data = [] 
    with open(path, "r", encoding="utf-8") as file: # specified encoding to ensure wikipedia file
        # can be read
        cleaned_data = file.read().replace(",", "").replace("\n", " ").replace(";", "").replace(":", "")
        data_by_sentences = cleaned_data.split(".") # assuming sentences are split by 
        # "." and not by new lines
    for sentence in data_by_sentences:
        docs_data.append([[word] for word in sentence.split(" ") if word != ''])
    S_data = utils.docs_to_indices(docs_data, word_to_num)
    return docs_data, S_data


def convert_to_lm_dataset(S):
    """
    Takes a dataset that is a list of sentences as an array of integer arrays.
    Returns the dataset a bigram prediction problem. For any word, predict the
    next work.
    IMPORTANT: we have two padding tokens at the beginning but since we are
    training a bigram model, only one will be used.
    """
    in_word_index, out_word_index = [], []
    for sentence in S:
        for i in range(len(sentence)):
            if i < 2:
                continue
            in_word_index.append(sentence[i - 1])
            out_word_index.append(sentence[i])
    return in_word_index, out_word_index


def shuffle_training_data(in_word_index, out_word_index):
    combined = list(zip(in_word_index, out_word_index))
    random.shuffle(combined)
    return list(zip(*combined))


def int_to_one_hot(number, dim):
    res = np.zeros(dim)
    res[number] = 1.0
    return res


def lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, params):

    data = np.zeros([BATCH_SIZE, input_dim])
    labels = np.zeros([BATCH_SIZE, output_dim])

    # Construct the data batch and run you backpropogation implementation
    ### YOUR CODE HERE
    input_indices = np.random.choice(len(in_word_index), size=BATCH_SIZE)
    input_words = in_word_index_np[input_indices]
    for i, index in enumerate(input_indices):
        data[i] = num_to_word_embedding[input_words[i]]
        labels[i] = int_to_one_hot(out_word_index[index], output_dim)
    cost, grad = forward_backward_prop(data, labels, params, dimensions)
    ### END YOUR CODE

    cost /= BATCH_SIZE
    grad /= BATCH_SIZE
    return cost, grad


def eval_neural_lm(eval_data_path):
    """
    Evaluate perplexity (use dev set when tuning and test at the end)
    """
    _, S_dev = load_data_as_sentences(eval_data_path, word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_dev)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    perplexity = 0
    ### YOUR CODE HERE
    for i in range(num_of_examples):
        input_word = num_to_word_embedding[in_word_index[i]]
        perplexity += np.log2(forward(input_word, out_word_index[i], params, dimensions))
    perplexity /= num_of_examples
    perplexity = 2 ** (-perplexity)
    ### END YOUR CODE

    return perplexity


def eval_neural_lm_q3c(eval_data_path):
    """
    This function is used in q3c only!
    It is identical to the original eval_neural_lm function, but uses the q3d load function in order
    to deal with the format of shakespeare and wikipedia texts. 
    Evaluate perplexity (use dev set when tuning and test at the end)
    """
    _, S_dev = load_data_as_sentences_q3c(eval_data_path, word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_dev)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    perplexity = 0
    ### YOUR CODE HERE
    for i in range(num_of_examples):
        input_word = num_to_word_embedding[in_word_index[i]]
        perplexity += np.log2(forward(input_word, out_word_index[i], params, dimensions))
    perplexity /= num_of_examples
    perplexity = 2 ** (-perplexity)
    ### END YOUR CODE

    return perplexity


if __name__ == "__main__":
    # Load the vocabulary
    # "C:/Users/Maya/NLP_HW2/NLP-HW2/HW2/data/lm/vocab.ptb.txt"
    # 
    # if os.path.exists("wikipedia_for_perplexity.txt"):
    #     print("wiki exists!")
    # if os.path.exists("shakespeare_for_perplexity.txt"):
    #     print("shakes exists!")

    vocab = pd.read_table("data/lm/vocab.ptb.txt",
                          header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
    

    vocabsize = 2000
    num_to_word = dict(enumerate(vocab.index[:vocabsize]))
    num_to_word_embedding = load_vocab_embeddings()
    word_to_num = utils.invert_dict(num_to_word)

    # Load the training data
    _, S_train = load_data_as_sentences('data/lm/ptb-train.txt', word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_train)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    random.seed(31415)
    np.random.seed(9265)
    in_word_index, out_word_index = shuffle_training_data(in_word_index, out_word_index)
    in_word_index_np = np.array(in_word_index)
    startTime = time.time()

    # Training should happen here
    # Initialize parameters randomly
    # Construct the params
    input_dim = 50
    hidden_dim = 50
    output_dim = vocabsize
    dimensions = [input_dim, hidden_dim, output_dim]
    params = np.random.randn((input_dim + 1) * hidden_dim + (
        hidden_dim + 1) * output_dim, )
    print(f"#params: {len(params)}")
    print(f"#train examples: {num_of_examples}")

    # run SGD
    params = sgd(
            lambda vec: lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, vec),
            params, LEARNING_RATE, NUM_OF_SGD_ITERATIONS, None, True, 1000)

    print(f"training took {time.time() - startTime} seconds")

    # Evaluate perplexity with dev-data
    perplexity = eval_neural_lm('data/lm/ptb-dev.txt')
    print(f"dev perplexity : {perplexity}")

    # Evaluate perplexity with test-data for shakespeare - no PP
    if os.path.exists('shakespeare_for_perplexity.txt'):
        perplexity = eval_neural_lm('shakespeare_for_perplexity.txt')
        print(f"shakespeare test perplexity without PP: {perplexity}")
    if os.path.exists("wikipedia_for_perplexity.txt"):
        perplexity = eval_neural_lm("wikipedia_for_perplexity.txt")
        print(f"wikipedia test perplexity without PP: {perplexity}")


    ### Q3D ###

    # Evaluate perplexity with test-data for shakespeare - after PP
    if os.path.exists('shakespeare_for_perplexity.txt'):
        perplexity = eval_neural_lm_q3c('shakespeare_for_perplexity.txt')
        print(f"shakespeare test perplexity after PP: {perplexity}")
    else:
        print("test perplexity will be evaluated only at test time!")
    # Evaluate perplexity with test-data for wikipedia - after PP
    if os.path.exists("wikipedia_for_perplexity.txt"):
        perplexity = eval_neural_lm_q3c("wikipedia_for_perplexity.txt")
        print(f"wikipedia test perplexity after PP: {perplexity}")
    else:
        print("test perplexity will be evaluated only at test time!")
    #next step: run with the original load func like Noa said in forum. 
    # then, edit my answer to 3c and 3d according to it. 
