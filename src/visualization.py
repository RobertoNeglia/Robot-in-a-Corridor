"""
Visualization utilities for the robot corridor environment.
"""

import glob
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from IPython.display import clear_output


def plot_training_progress(corridor, episode, steps, reward, epsilon):
    """
    Display the current state of the corridor during training with statistics.

    Args:
        corridor: Corridor environment instance
        episode: Current episode number
        steps: Number of steps taken in the last episode
        reward: Total reward obtained in the last episode
        epsilon: Current exploration rate
    """
    time.sleep(0.3)
    clear_output(wait=True)
    corridor.plot_corridor()
    plt.show()
    print(
        f"Episode: {episode}\n"
        f"Last number of steps: {steps}\n"
        f"Last total reward: {reward}\n"
        f"Epsilon: {epsilon}"
    )


def plot_training_results(reward_history, steps_history):
    """
    Plot the training results showing reward and steps over episodes.

    Args:
        reward_history: List of total rewards per episode
        steps_history: List of steps taken per episode
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(reward_history)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Reward per Episode")
    ax1.grid(True)

    ax2.plot(steps_history)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Steps per Episode")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_algorithm_comparison(results, metric="rewards", window=50):
    """
    Plot comparison of multiple algorithms.

    Args:
        results: Dictionary from compare_algorithms function
        metric: 'rewards' or 'steps'
        window: Window size for moving average smoothing
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))

    for algo_name, data in results.items():
        if metric == "rewards":
            mean_values = data["mean_rewards"]
            std_values = data["std_rewards"]
            ylabel = "Total Reward"
        else:
            mean_values = data["mean_steps"]
            std_values = data["std_steps"]
            ylabel = "Steps"

        # Apply moving average for smoothing
        if window > 1:
            mean_smooth = np.convolve(
                mean_values, np.ones(window) / window, mode="valid"
            )
            std_smooth = np.convolve(std_values, np.ones(window) / window, mode="valid")
            x = np.arange(window - 1, len(mean_values))
        else:
            mean_smooth = mean_values
            std_smooth = std_values
            x = np.arange(len(mean_values))

        # Plot mean with confidence interval
        ax.plot(x, mean_smooth, label=algo_name, linewidth=2)
        ax.fill_between(
            x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.2
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Algorithm Comparison: {ylabel} Over Episodes", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print final performance statistics
    print("\nFinal Performance (last 100 episodes average):")
    print("-" * 60)
    for algo_name, data in results.items():
        mean = data["final_performance_mean"]
        std = data["final_performance_std"]
        print(f"{algo_name:20s}: {mean:8.2f} ± {std:6.2f}")
    print("-" * 60)


def save_corridor_frame(corridor, output_dir, frame_number, dpi=300):
    """
    Save a single frame of the corridor as an image.

    Args:
        corridor: Corridor environment instance
        output_dir: Directory to save the image
        frame_number: Frame number for filename
        dpi: Resolution of the saved image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"frame-{str(frame_number).zfill(3)}.jpg"
    filepath = output_path / filename

    corridor.plot_corridor()
    plt.savefig(
        filepath,
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )
    plt.show()


def generate_episode_gif(
    corridor,
    agent,
    output_path,
    show_frames=True,
    frame_interval=100,
    dpi=300,
    max_frames=500,
):
    """
    Generate a GIF animation of the agent navigating the corridor.

    Args:
        corridor: Corridor environment instance
        agent: Agent instance with trained policy
        output_path: Path to save the GIF file
        show_frames: Whether to display frames during generation
        frame_interval: Time between frames in milliseconds
        dpi: Resolution of the frames
        max_frames: Maximum number of frames to prevent infinite loops

    Returns:
        Number of frames generated
    """
    # Create temporary directory for frames
    # if it already exists, clean it up

    temp_dir = Path(output_path).parent / "temp_frames"
    if temp_dir.exists():
        for frame_file in temp_dir.glob("*.jpg"):
            frame_file.unlink()
    else:
        temp_dir.mkdir(parents=True, exist_ok=True)

    corridor.reset_env()
    frame_count = 0

    # Generate frames
    while not corridor.is_terminal() and frame_count < max_frames:
        if show_frames:
            time.sleep(0.1)
            clear_output(wait=True)

        # Save frame
        corridor.plot_corridor()
        frame_path = temp_dir / f"frame-{str(frame_count).zfill(3)}.jpg"
        plt.savefig(
            frame_path,
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )
        if show_frames:
            plt.show()
        else:
            plt.close()

        # Take action
        state = corridor.get_state()
        allowed_actions = corridor.get_allowed_actions(state)
        action = agent.greedy_action(state, allowed_actions)
        corridor.update(action)
        frame_count += 1

    # Save final frame
    if show_frames:
        clear_output(wait=True)
    corridor.plot_corridor()
    frame_path = temp_dir / f"frame-{str(frame_count).zfill(3)}.jpg"
    plt.savefig(
        frame_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )
    if show_frames:
        plt.show()
    else:
        plt.close()
    frame_count += 1

    # Create GIF from frames
    create_gif_from_images(temp_dir, output_path, frame_interval)

    # Clean up temporary frames
    for frame_file in temp_dir.glob("*.jpg"):
        frame_file.unlink()
    temp_dir.rmdir()

    return frame_count


def create_gif_from_images(image_dir, output_path, frame_interval=100):
    """
    Create an animated GIF from a directory of images.

    Args:
        image_dir: Directory containing image files
        output_path: Path to save the GIF file
        frame_interval: Time between frames in milliseconds
    """
    # Load all images
    image_pattern = str(Path(image_dir) / "*.jpg")
    image_files = sorted(glob.glob(image_pattern))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    images_array = [Image.open(img) for img in image_files]

    # Create animation
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(10, 3)
    im = ax.imshow(images_array[0], animated=True)
    plt.axis("off")

    def update(i):
        im.set_array(images_array[i])
        return (im,)

    animation_fig = animation.FuncAnimation(
        fig,
        update,
        frames=len(images_array),
        interval=frame_interval,
        blit=True,
        repeat_delay=100,
    )

    # Save GIF
    animation_fig.save(output_path, writer="pillow")
    plt.close()
    print(f"GIF saved to {output_path} ({len(images_array)} frames)")
