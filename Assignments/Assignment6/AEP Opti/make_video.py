# pyrefly: ignore [missing-import]
import cv2
import os
import glob
import argparse
import re
from tqdm import tqdm

def natural_sort_key(s):
    """Sort strings with numbers naturally (e.g., file_2.png comes before file_10.png)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def create_video(image_folder, output_video, fps=2):
    # Grab all PNG and JPG images in the folder
    search_paths = [os.path.join(image_folder, "*.png"), os.path.join(image_folder, "*.jpg")]
    images = []
    for path in search_paths:
        images.extend(glob.glob(path))
        
    if not images:
        print(f"Error: No .png or .jpg images found in folder '{image_folder}'.")
        return
        
    # Sort images naturally based on their filenames
    images.sort(key=natural_sort_key)
    
    # Read the first image to get the dimensions
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Error: Could not read image {images[0]}")
        return
        
    height, width, layers = frame.shape

    # Setup the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Output as MP4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Found {len(images)} images.")
    print(f"Generating video '{output_video}' at {fps} FPS...")
    
    for image_path in tqdm(images, desc="Generating video", colour="green"):
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)

    # Cleanup
    video.release()
    cv2.destroyAllWindows()
    print("Done! Video saved successfully.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Convert images in a folder to a timelapse video.")
    # parser.add_argument("db", type=str, help="Path to the folder containing images (e.g., 'plots/best_15')")
    # parser.add_argument("--fps", type=float, default=2.0, help="Frames per second (default: 2.0)")
    # parser.add_argument("--out", type=str, default="timelapse.mp4", help="Output video file name (default: timelapse.mp4)")
    
    # args = parser.parse_args()
    
    db = "/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/plots/BAR_BAU_LSP_3.25MW/14"
    out = "/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/optimized_layout_n14_t2.mp4"
    fps = 2.0

    create_video(db, out, fps)
