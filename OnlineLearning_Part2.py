# Homework 2, Part 2: Online Learning

# In this homework, you will implement an online learning algorithm for a beta-binomial model.

# BAIM Mohamed Jalal


def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def binomial_coefficient(n, k):
    if k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

# Function to compute the likelihood based on MLE
def compute_likelihood(n, k):
    if n == 0:
        return 0
    p_hat = k / n
    coeff = binomial_coefficient(n, k)
    likelihood = coeff * (p_hat ** k) * ((1 - p_hat) ** (n - k))
    return likelihood

def update_beta_prior(a, b, k, n):
    a_new = a + k
    b_new = b + n - k
    return a_new, b_new

def read_data(file_path):
    data_lines = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip() 
                if not line:
                    continue 
                case_number = len(data_lines) + 1 
                data_lines.append((case_number, line))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return data_lines

def beta_binomial_online_learning(file_path, a_init, b_init):
    a = a_init
    b = b_init
    data_lines = read_data(file_path)

    if not data_lines:
        print("No data found in the file.")
        return

    for idx, (case_number, data_str) in enumerate(data_lines, 1):
        data = [int(bit) for bit in data_str if bit in '01']
        n = len(data)
        k = sum(data)

        if n == 0:
            print(f"Case {case_number}: {data_str}")
            print("No data available.\n")
            continue

        likelihood = compute_likelihood(n, k)

        print(f"Case {case_number}: {data_str}")
        print(f"Number of trials (n): {n}")
        print(f"Number of successes (k): {k}")
        print(f"Likelihood: {likelihood:.6f}")
        print(f"Beta prior: a = {a}, b = {b}")

        a, b = update_beta_prior(a, b, k, n)

        print(f"Beta posterior: a = {a}, b = {b}\n")

# Main function
def main():
    file_path = 'test_file.txt'
    a_init = int(input("Enter the initial value of a: "))
    b_init = int(input("Enter the initial value of b: "))

    beta_binomial_online_learning(file_path, a_init, b_init)


if __name__ == "__main__":
    main()