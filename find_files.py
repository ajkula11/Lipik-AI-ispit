import tkinter as tk
from tkinter import filedialog


def select_files():
    """
    Open file dialogs to select video and offset files.

    Returns:
        tuple: (video_left, video_right, offset_left, offset_right)
    """
    # Create root window and hide it
    root = tk.Tk()
    # root.withdraw()

    # Prompt for video files
    print("Select left video file...")
    video_left = filedialog.askopenfilename(
        title="Select left video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not video_left:
        print("No left video selected. Exiting.")
        exit(1)

    print("Select right video file...")
    video_right = filedialog.askopenfilename(
        title="Select right video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not video_right:
        print("No right video selected. Exiting.")
        exit(1)

    # Prompt for offset files
    print("Select left offset file...")
    offset_left = filedialog.askopenfilename(
        title="Select left offset file",
        filetypes=[("Text files", "*.txt"), ("Offset files", "*.offset"), ("All files", "*.*")]
    )
    if not offset_left:
        print("No left offset file selected. Exiting.")
        exit(1)

    print("Select right offset file...")
    offset_right = filedialog.askopenfilename(
        title="Select right offset file",
        filetypes=[("Text files", "*.txt"), ("Offset files", "*.offset"), ("All files", "*.*")]
    )
    if not offset_right:
        print("No right offset file selected. Exiting.")
        exit(1)

    root.destroy()

    return video_left, video_right, offset_left, offset_right


def read_offset_file(file_path):
    """
    Read offset value from a text file.

    Args:
        file_path: Path to the text file containing offset value

    Returns:
        float: The offset value in seconds

    Raises:
        ValueError: If file content cannot be converted to float
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()

    try:
        offset = float(content)
        return offset
    except ValueError:
        raise ValueError(f"File content '{content}' cannot be converted to float")
        