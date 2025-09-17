from datasets import load_dataset

def main():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    dataset.save_to_disk("data/raw")

if __name__ == "__main__":
    main()
