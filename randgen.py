import random

def main():
    print("Generating a 50 million random numbers. Please be patient.")
    with open('random_numbers.txt', 'w') as file:
        for _ in range(50_000_000):
            number = random.uniform(0, 1000)
            file.write(str(number) + '\n')

if __name__ == "__main__":
    main()