import pandas as pd
import matplotlib.pyplot as plt

def plot_from_csv(file_path, save_path, n):
    """
    Reads data from a CSV file, extracts x, y_test, and y_pred, and plots y_test and y_pred together.

    Args:
        file_path (str): Path to the CSV file.
        save_path (str): Path to save the resulting plot.
        n (int): Number of columns to use for x.
    """
    # Step 1: Read the CSV file
    data = pd.read_csv(file_path)
    
    # Step 2: Extract x, y_test, and y_pred
    if data.shape[1] < n + 2:
        raise ValueError("Insufficient columns in the CSV file for the specified 'n'.")
    x = data.iloc[:, :n]  # First n columns for x
    y_test = data.iloc[:, -2]  # Second last column for y_test
    y_pred = data.iloc[:, -1]  # Last column for y_pred

    # Step 3: Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.index, y_test, label='y_test', color='blue', linewidth=2)
    plt.plot(x.index, y_pred, label='y_pred', color='red', linewidth=2, linestyle='--')
    plt.title("Comparison of y_test and y_pred")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)

    # Step 4: Save the plot
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Graph saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Replace 'result.csv' with your CSV file path and 'graph.png' with your desired output file name
    plot_from_csv("result.csv", "graph.png", n=5)
