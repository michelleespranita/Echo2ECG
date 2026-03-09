import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def play_video(video, save_path=None):
    """Plays a video.

    Args:
        video: A np.ndarray of shape (T, H, W, C)
    """
    if len(video.shape) == 3:
        print(f'Video cannot be played! Shape: {video.shape}')
        plt.imshow(video)
        return

    fig, ax = plt.subplots()
    im = ax.imshow(video[0], cmap="gray")  # Display first frame in grayscale

    # Update function for animation
    def update(frame):
        im.set_array(video[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(video), interval=30)

    if save_path and save_path.endswith('.mp4'): # save as .mp4
        ani.save(save_path, writer='ffmpeg')
        print(f'Video saved to {save_path}')
    elif save_path and not save_path.endswith('.mp4'):
        print('Video not saved because video format is not supported!')

    plt.close(fig)

    return HTML(ani.to_jshtml())