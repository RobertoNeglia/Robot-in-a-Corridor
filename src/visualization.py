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


def visualize_training_progress_overlay(
    env_config,
    AgentClass,
    agent_params,
    episodes_to_show=None,
    n_epochs=1000,
    max_steps=500,
    output_path="images/training_progress_overlay.png",
):
    """
    Visualize multiple agents from different training stages overlaid on the same image.
    Each agent is shown in a different color with decreasing opacity for earlier episodes.

    Args:
        env_config: Dictionary with environment configuration
        AgentClass: Agent class to train
        agent_params: Dictionary with agent parameters
        episodes_to_show: List of episode numbers to visualize
        n_epochs: Total number of training episodes
        max_steps: Maximum steps per episode
        output_path: Path to save the output image

    Returns:
        Dictionary with training statistics
    """
    import numpy as np
    from .environment import Corridor
    from .training import train_td_learning
    from .agent import MonteCarloAgent, SARSAAgent

    print("=" * 70)
    print("MULTI-AGENT OVERLAY VISUALIZATION")
    print("=" * 70)

    # Auto-select episodes
    if episodes_to_show is None:
        episodes_to_show = [
            0,
            n_epochs // 4,
            n_epochs // 2,
            3 * n_epochs // 4,
            n_epochs - 1,
        ]

    print(f"Training and capturing agents at episodes: {episodes_to_show}")

    # Train and capture snapshots
    env = Corridor(**env_config)
    agent = AgentClass(env, **agent_params)

    snapshots = {}
    steps_history = []
    reward_history = []

    # Training loop
    for episode in range(n_epochs):
        env.reset_env()
        agent.reset_episode()

        state = env.get_state()

        if isinstance(agent, SARSAAgent):
            action = agent.choose_action(state)

        while not env.is_terminal():
            if not isinstance(agent, SARSAAgent):
                action = agent.choose_action(state)

            env.update(action)
            next_state, reward = env.get_state_and_reward()

            if isinstance(agent, MonteCarloAgent):
                agent.update_trajectory(next_state, reward)
            elif isinstance(agent, SARSAAgent):
                if not env.is_terminal():
                    next_action = agent.choose_action(next_state)
                else:
                    next_action = None
                agent.update(state, action, reward, next_state, next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state)

            state = next_state

            if agent.steps > max_steps:
                break

        reward_history.append(agent.total_reward)
        steps_history.append(agent.steps)

        if isinstance(agent, MonteCarloAgent):
            agent.update()
        else:
            agent.decay_epsilon()

        # Capture snapshot
        if episode in episodes_to_show:
            print(
                f"  Episode {episode}: reward={agent.total_reward:.0f}, steps={agent.steps}"
            )

            test_env = Corridor(**env_config)
            test_path = []
            test_steps = 0
            max_test_steps = test_env.length * 3

            saved_epsilon = agent.epsilon
            agent.epsilon = 0.0

            while not test_env.is_terminal() and test_steps < max_test_steps:
                test_state = test_env.get_state()
                test_path.append((test_state[1], test_state[0], test_state[2]))
                allowed_actions = test_env.get_allowed_actions(test_state)
                test_action = agent.greedy_action(test_state, allowed_actions)
                test_env.update(test_action)
                test_steps += 1

            agent.epsilon = saved_epsilon

            snapshots[episode] = {
                "path": test_path,
                "reached_goal": test_env.is_terminal(),
                "steps": test_steps,
                "avg_reward": (
                    np.mean(reward_history[-50:])
                    if len(reward_history) >= 50
                    else np.mean(reward_history)
                ),
            }

    print(f"\nTraining complete! Final reward: {np.mean(reward_history[-100:]):.2f}")

    # Create overlay visualization
    print("\nGenerating overlay visualization...")

    # Create base corridor image
    base_env = Corridor(**env_config)

    # Use a color map for different episodes
    colors = plt.cm.viridis(np.linspace(0, 1, len(episodes_to_show)))

    fig, ax = plt.subplots(figsize=(20, 4))

    # Draw base corridor (walls and goal)
    from .utils import SYMBOLS_TO_VALUES

    corridor_display = base_env.corridor.copy()
    corridor_display[:] = " "
    corridor_display[:, 0] = "#"
    corridor_display[0, :] = "#"
    corridor_display[base_env.width - 1, :] = "#"
    corridor_display[1 : base_env.width - 1, base_env.length - 1] = "G"

    res = np.vectorize(SYMBOLS_TO_VALUES.get)(corridor_display)
    ax.imshow(res, cmap="gray", interpolation="nearest", alpha=0.3)

    # Overlay each agent's path
    for idx, episode in enumerate(episodes_to_show):
        snapshot = snapshots[episode]
        path = snapshot["path"]

        if len(path) > 0:
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]

            # Calculate alpha based on episode (later = more opaque)
            alpha = 0.3 + (idx / len(episodes_to_show)) * 0.7

            # Plot the path
            color = colors[idx]
            ax.plot(
                x_coords,
                y_coords,
                "o-",
                color=color,
                alpha=alpha,
                linewidth=2,
                markersize=4,
                label=f"Episode {episode}",
            )

            # Mark start with a larger marker
            ax.plot(
                x_coords[0],
                y_coords[0],
                "o",
                color=color,
                markersize=12,
                alpha=alpha,
                markeredgecolor="white",
                markeredgewidth=1,
            )

            # Mark end differently if goal reached
            if snapshot["reached_goal"]:
                ax.plot(
                    x_coords[-1],
                    y_coords[-1],
                    "*",
                    color=color,
                    markersize=20,
                    alpha=alpha,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                )

    ax.set_xlim(-0.5, base_env.length - 0.5)
    ax.set_ylim(base_env.width - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(
        f"Agent Behavior Evolution Over Training\n"
        f"{AgentClass.__name__} on {base_env.length-3}-unit Corridor",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Overlay visualization saved to '{output_path}'")

    return {
        "steps_history": steps_history,
        "reward_history": reward_history,
        "snapshots": snapshots,
    }


def visualize_training_progress_animated_overlay(
    env_config,
    AgentClass,
    agent_params,
    episodes_to_show=None,
    n_epochs=1000,
    max_steps=500,
    output_path="images/training_progress_animated.gif",
):
    """
    Create an animated GIF showing agents from different training stages appearing progressively.
    Each new agent is added with its own color, showing the evolution of behavior.

    Args:
        env_config: Dictionary with environment configuration
        AgentClass: Agent class to train
        agent_params: Dictionary with agent parameters
        episodes_to_show: List of episode numbers to visualize
        n_epochs: Total number of training episodes
        max_steps: Maximum steps per episode
        output_path: Path to save the output GIF

    Returns:
        Dictionary with training statistics
    """
    import numpy as np
    from PIL import Image
    from .environment import Corridor
    from .training import train_td_learning
    from .agent import MonteCarloAgent, SARSAAgent

    print("=" * 70)
    print("ANIMATED MULTI-AGENT OVERLAY VISUALIZATION")
    print("=" * 70)

    # Auto-select episodes
    if episodes_to_show is None:
        episodes_to_show = [
            0,
            n_epochs // 4,
            n_epochs // 2,
            3 * n_epochs // 4,
            n_epochs - 1,
        ]

    print(f"Training and capturing agents at episodes: {episodes_to_show}")

    # Train and capture snapshots
    env = Corridor(**env_config)
    agent = AgentClass(env, **agent_params)

    snapshots = {}
    steps_history = []
    reward_history = []

    # Training loop (same as overlay version)
    for episode in range(n_epochs):
        env.reset_env()
        agent.reset_episode()

        state = env.get_state()

        if isinstance(agent, SARSAAgent):
            action = agent.choose_action(state)

        while not env.is_terminal():
            if not isinstance(agent, SARSAAgent):
                action = agent.choose_action(state)

            env.update(action)
            next_state, reward = env.get_state_and_reward()

            if isinstance(agent, MonteCarloAgent):
                agent.update_trajectory(next_state, reward)
            elif isinstance(agent, SARSAAgent):
                if not env.is_terminal():
                    next_action = agent.choose_action(next_state)
                else:
                    next_action = None
                agent.update(state, action, reward, next_state, next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state)

            state = next_state

            if agent.steps > max_steps:
                break

        reward_history.append(agent.total_reward)
        steps_history.append(agent.steps)

        if isinstance(agent, MonteCarloAgent):
            agent.update()
        else:
            agent.decay_epsilon()

        # Capture snapshot
        if episode in episodes_to_show:
            print(
                f"  Episode {episode}: reward={agent.total_reward:.0f}, steps={agent.steps}"
            )

            test_env = Corridor(**env_config)
            test_path = []
            test_steps = 0
            max_test_steps = test_env.length * 3

            saved_epsilon = agent.epsilon
            agent.epsilon = 0.0

            while not test_env.is_terminal() and test_steps < max_test_steps:
                test_state = test_env.get_state()
                test_path.append((test_state[1], test_state[0], test_state[2]))
                allowed_actions = test_env.get_allowed_actions(test_state)
                test_action = agent.greedy_action(test_state, allowed_actions)
                test_env.update(test_action)
                test_steps += 1

            agent.epsilon = saved_epsilon

            snapshots[episode] = {
                "path": test_path,
                "reached_goal": test_env.is_terminal(),
                "steps": test_steps,
                "avg_reward": (
                    np.mean(reward_history[-50:])
                    if len(reward_history) >= 50
                    else np.mean(reward_history)
                ),
            }

    print(f"\nTraining complete! Final reward: {np.mean(reward_history[-100:]):.2f}")

    # Create animated overlay
    print("\nGenerating animated overlay...")

    base_env = Corridor(**env_config)
    colors = plt.cm.viridis(np.linspace(0, 1, len(episodes_to_show)))

    frames = []

    # Create frames progressively adding each agent
    for num_agents in range(1, len(episodes_to_show) + 1):
        fig, ax = plt.subplots(figsize=(20, 4))

        # Draw base corridor
        from .utils import SYMBOLS_TO_VALUES

        corridor_display = base_env.corridor.copy()
        corridor_display[:] = " "
        corridor_display[:, 0] = "#"
        corridor_display[0, :] = "#"
        corridor_display[base_env.width - 1, :] = "#"
        corridor_display[1 : base_env.width - 1, base_env.length - 1] = "G"

        res = np.vectorize(SYMBOLS_TO_VALUES.get)(corridor_display)
        ax.imshow(res, cmap="gray", interpolation="nearest", alpha=0.3)

        # Overlay agents up to current number
        for idx in range(num_agents):
            episode = episodes_to_show[idx]
            snapshot = snapshots[episode]
            path = snapshot["path"]

            if len(path) > 0:
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]

                # Calculate alpha
                alpha = 0.3 + (idx / len(episodes_to_show)) * 0.7

                color = colors[idx]
                ax.plot(
                    x_coords,
                    y_coords,
                    "o-",
                    color=color,
                    alpha=alpha,
                    linewidth=2,
                    markersize=4,
                    label=f"Episode {episode}",
                )

                ax.plot(
                    x_coords[0],
                    y_coords[0],
                    "o",
                    color=color,
                    markersize=12,
                    alpha=alpha,
                    markeredgecolor="white",
                    markeredgewidth=1,
                )

                if snapshot["reached_goal"]:
                    ax.plot(
                        x_coords[-1],
                        y_coords[-1],
                        "*",
                        color=color,
                        markersize=20,
                        alpha=alpha,
                        markeredgecolor="white",
                        markeredgewidth=1.5,
                    )

        ax.set_xlim(-0.5, base_env.length - 0.5)
        ax.set_ylim(base_env.width - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"Agent Behavior Evolution Over Training (Agents 1-{num_agents})\n"
            f"{AgentClass.__name__} on {base_env.length-3}-unit Corridor",
            fontsize=14,
            pad=15,
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_array = image_array.reshape(height, width, 4)
        image = Image.fromarray(image_array[:, :, :3])  # Drop alpha channel
        frames.append(image)
        plt.close()

    # Hold final frame longer
    for _ in range(10):
        frames.append(frames[-1])

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=800,  # 800ms per frame
        loop=0,
    )

    print(f"✓ Animated overlay saved to '{output_path}' ({len(frames)} frames)")

    return {
        "steps_history": steps_history,
        "reward_history": reward_history,
        "snapshots": snapshots,
    }


def visualize_training_progress(
    env_config,
    AgentClass,
    agent_params,
    episodes_to_show=None,
    n_epochs=1000,
    max_steps=500,
    output_path="images/training_progress.gif",
):
    """
    Visualize how an agent's behavior changes during training by capturing
    snapshots at different training stages.

    Args:
        env_config: Dictionary with environment configuration (e.g., {'length': 100, 'width': 5})
        AgentClass: Agent class to train
        agent_params: Dictionary with agent parameters (e.g., {'gamma': 0.9, 'alpha': 0.5, 'epsilon': 0.3})
        episodes_to_show: List of episode numbers to visualize (e.g., [0, 100, 500, 1000])
                         If None, will auto-select evenly spaced episodes
        n_epochs: Total number of training episodes
        max_steps: Maximum steps per episode
        output_path: Path to save the output GIF

    Returns:
        Dictionary with training statistics
    """
    import numpy as np
    from .environment import Corridor
    from .training import train_td_learning
    from .agent import MonteCarloAgent

    print("=" * 70)
    print("VISUALIZING TRAINING PROGRESS")
    print("=" * 70)

    # Auto-select episodes to show if not provided
    if episodes_to_show is None:
        episodes_to_show = [
            0,
            n_epochs // 4,
            n_epochs // 2,
            3 * n_epochs // 4,
            n_epochs - 1,
        ]

    print(f"Will capture agent behavior at episodes: {episodes_to_show}")
    print(f"Training for {n_epochs} episodes...")

    # Create environment and agent
    env = Corridor(**env_config)
    agent = AgentClass(env, **agent_params)

    # Store snapshots of Q/V functions at key episodes
    snapshots = {}
    steps_history = []
    reward_history = []

    # Training loop with snapshots
    for episode in range(n_epochs):
        env.reset_env()
        agent.reset_episode()

        state = env.get_state()

        # For SARSA
        from .agent import SARSAAgent

        if isinstance(agent, SARSAAgent):
            action = agent.choose_action(state)

        # Run episode
        while not env.is_terminal():
            if not isinstance(agent, SARSAAgent):
                action = agent.choose_action(state)

            env.update(action)
            next_state, reward = env.get_state_and_reward()

            # Update based on agent type
            if isinstance(agent, MonteCarloAgent):
                agent.update_trajectory(next_state, reward)
            elif isinstance(agent, SARSAAgent):
                if not env.is_terminal():
                    next_action = agent.choose_action(next_state)
                else:
                    next_action = None
                agent.update(state, action, reward, next_state, next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state)

            state = next_state

            if agent.steps > max_steps:
                break

        reward_history.append(agent.total_reward)
        steps_history.append(agent.steps)

        # Update for Monte Carlo
        if isinstance(agent, MonteCarloAgent):
            agent.update()
        else:
            agent.decay_epsilon()

        # Capture snapshot at key episodes
        if episode in episodes_to_show:
            print(
                f"  Episode {episode}: Capturing snapshot (reward={agent.total_reward:.0f}, steps={agent.steps})"
            )

            # Test the current policy
            test_env = Corridor(**env_config)
            test_path = []
            test_steps = 0
            # Allow more steps for testing, based on corridor length
            max_test_steps = test_env.length * 3

            # Use greedy policy (no exploration) for visualization
            saved_epsilon = agent.epsilon
            agent.epsilon = 0.0

            while not test_env.is_terminal() and test_steps < max_test_steps:
                test_state = test_env.get_state()
                test_path.append(
                    (test_state[1], test_state[0], test_state[2])
                )  # (x, y, orientation)
                allowed_actions = test_env.get_allowed_actions(test_state)
                test_action = agent.greedy_action(test_state, allowed_actions)
                test_env.update(test_action)
                test_steps += 1

            # Restore epsilon
            agent.epsilon = saved_epsilon

            snapshots[episode] = {
                "path": test_path,
                "reached_goal": test_env.is_terminal(),
                "steps": test_steps,
                "epsilon": agent.epsilon,
                "avg_reward_last_50": (
                    np.mean(reward_history[-50:])
                    if len(reward_history) >= 50
                    else np.mean(reward_history)
                ),
            }

    print("\nTraining complete!")
    print(f"Final performance: {np.mean(reward_history[-100:]):.2f} avg reward")

    # Create visualization
    print("\nGenerating multi-episode visualization...")

    # Create temporary directory
    temp_dir = Path(output_path).parent / "temp_multi_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    # For each snapshot, create frames showing the agent's path
    for episode in episodes_to_show:
        snapshot = snapshots[episode]
        path = snapshot["path"]

        # Create frames for this episode's path
        for step_idx, (x, y, orientation) in enumerate(path):
            # Create a new environment and set position
            vis_env = Corridor(**env_config)
            vis_env.agent_pos = np.array([y, x])
            vis_env.agent_orientation = orientation

            # Update corridor display
            vis_env.corridor[:] = " "
            vis_env.corridor[:, 0] = "#"
            vis_env.corridor[0, :] = "#"
            vis_env.corridor[vis_env.width - 1, :] = "#"
            vis_env.corridor[1 : vis_env.width - 1, vis_env.length - 1] = "G"
            vis_env.corridor[y, x] = "X"

            # Plot
            fig, ax = plt.subplots(figsize=(15, 3))
            from .utils import SYMBOLS_TO_VALUES

            res = np.vectorize(SYMBOLS_TO_VALUES.get)(vis_env.corridor)
            ax.imshow(res, cmap=plt.cm.gray, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            # Add title with episode info
            title = (
                f"Episode {episode} | Step {step_idx}/{len(path)} | "
                f"ε={snapshot['epsilon']:.3f} | "
                f"Avg Reward (last 50 eps): {snapshot['avg_reward_last_50']:.1f}"
            )
            if snapshot["reached_goal"] and step_idx == len(path) - 1:
                title += " | ✓ GOAL!"
            ax.set_title(title, fontsize=12, pad=10)

            # Save frame
            frame_path = temp_dir / f"frame-{str(frame_count).zfill(4)}.jpg"
            plt.savefig(frame_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
            plt.close()
            frame_count += 1

        # Hold on the final frame of each episode
        for _ in range(10):
            frame_path = temp_dir / f"frame-{str(frame_count).zfill(4)}.jpg"
            # Copy the last frame
            import shutil

            last_frame = temp_dir / f"frame-{str(frame_count - 1).zfill(4)}.jpg"
            shutil.copy(last_frame, frame_path)
            frame_count += 1

    # Create GIF from all frames
    image_pattern = str(temp_dir / "*.jpg")
    image_files = sorted(glob.glob(image_pattern))

    if image_files:
        images_array = [Image.open(img) for img in image_files]

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        fig.set_size_inches(15, 3)
        im = ax.imshow(images_array[0], animated=True)
        plt.axis("off")

        def update(i):
            im.set_array(images_array[i])
            return (im,)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(images_array),
            interval=150,
            blit=True,
            repeat_delay=1000,
        )

        anim.save(output_path, writer="pillow")
        plt.close()
        print(
            f"\n✓ Training progress GIF saved to '{output_path}' ({frame_count} frames)"
        )

    # Clean up temporary frames
    for frame_file in temp_dir.glob("*.jpg"):
        frame_file.unlink()
    temp_dir.rmdir()

    return {
        "steps_history": steps_history,
        "reward_history": reward_history,
        "snapshots": snapshots,
    }
