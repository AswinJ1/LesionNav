import time
import matplotlib.pyplot as plt

def simulate_ddqn_training(epoch):
    """Simulate the time required for DDQN training for a given number of epochs."""
    # Simulate time in seconds (this should be replaced with actual training time measurement)
    simulated_time = epoch * 0.1  # Assuming each epoch takes 0.1 minutes
    time.sleep(simulated_time * 60)  # Simulate the delay (comment out for real use)
    return simulated_time

def measure_training_times():
    epochs = [50, 100, 150, 200, 250, 300]
    epoch_times = []

    for epoch in epochs:
        start_time = time.time()
        simulate_ddqn_training(epoch)
        end_time = time.time()
        
        duration = (end_time - start_time) / 60  # Convert to minutes
        epoch_times.append(duration)
        print(f"Epoch: {epoch}, Time: {duration:.2f} minutes")

    return epochs, epoch_times

def plot_time_per_epoch(epochs, epoch_times):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_times, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (in minutes)')
    plt.title('Time per Epoch for DDQN')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    epochs, epoch_times = measure_training_times()
    plot_time_per_epoch(epochs, epoch_times)