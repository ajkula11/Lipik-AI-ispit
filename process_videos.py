import cv2
import math
from collections import deque
import numpy as np
import os

from detect_basketball2 import detect_basketball

LOOKAHEAD = 10

def get_best_detection(detections):
    if not detections:
        return None
    return max(detections, key=lambda d: d["confidence"])

def bbox_center(bbox : tuple[4]):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def choose_detection(detections, previous_bbox=None, max_distance=150):
    """Choose detection closest to previous_bbox if exists,
    else choose most confident detection.
    If no detections provided, return None.
    """

    if not detections:
        return None

    if previous_bbox is None:
        return max(detections, key=lambda d: d["confidence"])

    px, py = bbox_center(previous_bbox)

    best = min(
        detections,
        key=lambda d: math.dist(bbox_center(d["bbox"]), (px, py))
    )

    # ovo je opasno, jer će ostati zaključano ako imamo previše uzastopnih non-detekcija.
    # distance = math.dist(bbox_center(best["bbox"]), (px, py))
    # if distance > max_distance:
    #     return None

    return best

def process_videos(video_left_path, video_right_path, offset_left, offset_right, save_video_path=None):
    """
    Open two videos starting at their corresponding offsets.

    Args:
        video_left_path: Path to left video file
        video_right_path: Path to right video file
        offset_left: Time offset in seconds for left video
        offset_right: Time offset in seconds for right video
    """

    GOOGLE_COLAB = True if save_video_path else False  # if True, save annotated video to disk

    # Open video captures
    cap_left = cv2.VideoCapture(video_left_path)
    cap_right = cv2.VideoCapture(video_right_path)

    if not cap_left.isOpened():
        raise ValueError(f"Could not open left video: {video_left_path}")
    if not cap_right.isOpened():
        raise ValueError(f"Could not open right video: {video_right_path}")

    # Get video properties
    fps_left = cap_left.get(cv2.CAP_PROP_FPS)
    fps_right = cap_right.get(cv2.CAP_PROP_FPS)
    width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if GOOGLE_COLAB:
        if not os.path.isdir(os.path.dirname(save_video_path)):
            raise FileNotFoundError(f"Folder not found: {os.path.dirname(save_video_path)}")
        combined_width = width_left + width_right
        combined_height = max(height_left, height_right)
        writer = cv2.VideoWriter(
            save_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_left / 5,   # because you process every 5th frame
            (combined_width, combined_height)
        )

    # hardkodirani roi
    # left: (x1, y1, x2, y2)
    roi_left = (
        240,
        530,
        width_left - 140,
        height_left - 30
    )

    # right
    roi_right = (
        40,
        500,
        width_right - 340,
        height_right - 60
    )

    # Calculate starting frame numbers based on offsets
    start_frame_left = int(offset_left * fps_left)
    start_frame_right = int(offset_right * fps_right)

    print(f"Left video: {fps_left} fps, {width_left}x{height_left}, starting at {offset_left}s (frame {start_frame_left})")
    print(f"Right video: {fps_right} fps, {width_right}x{height_right}, starting at {offset_right}s (frame {start_frame_right})")

    # Calculate display size to fit both videos on 1920x1080 screen
    # Each video gets half the width (960px), scale height proportionally
    display_width = 720
    scale_left = display_width / width_left
    scale_right = display_width / width_right
    display_height_left = int(height_left * scale_left)
    display_height_right = int(height_right * scale_right)

    cap_left.set(cv2.CAP_PROP_POS_FRAMES, start_frame_left)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, start_frame_right)

    # Variables to store basketball locations
    basketball_location_left = None
    basketball_location_right = None

    frame_buffer = deque()
    active_camera = "left"
    windows_created = False

    if GOOGLE_COLAB:
        frame_counter = 0
        MAX_FRAMES = 100  # 2800 or whatever you want

    while True:
        # Read frames
        for _ in range(5):
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

        # cropping to match training
        # NOTE: iskoristit ćemo roi parametar u detect_basketball
        # hL, wL = frame_left.shape[:2]
        # left_crop = frame_left[530:hL-30, 240:wL-140]
        # hR, wR = frame_right.shape[:2]
        # right_crop = frame_right[500:hR-60, 40:wR-340]

        # Break if either video ends
        if not ret_left or not ret_right:
            print("End of video reached")
            break

        # Detect basketball on both frames, choose best detection, and save it (if exists) for next iteration
        annotated_left, detections_left = detect_basketball(frame_left, roi=roi_left)
        annotated_right, detections_right = detect_basketball(frame_right, roi=roi_right)
        chosen_frame_left = choose_detection(detections_left, basketball_location_left)
        chosen_frame_right = choose_detection(detections_right, basketball_location_right)
        if chosen_frame_left:
            basketball_location_left = chosen_frame_left["bbox"]
            print(f"Basketball at left video detected at: {basketball_location_left}")
        if chosen_frame_right:
            basketball_location_right = chosen_frame_right["bbox"]
            print(f"Basketball at right video detected at: {basketball_location_right}")

        # buffer for lookahead decision
        frame_buffer.append({
            "left_frame": annotated_left,
            "right_frame": annotated_right,
            "left_has_detection": chosen_frame_left is not None,
            "right_has_detection": chosen_frame_right is not None,
        })

        if len(frame_buffer) <= LOOKAHEAD:
            continue

        # # Save detection location if basketball is detected (choose one with highest confidence, if multiple detections)
        # best_left = get_best_detection(detections_left)
        # if best_left:
        #     basketball_location_left = best_left["bbox"]
        #     print(f"Basketball at left video detected at: {basketball_location_left}")

        # best_right = get_best_detection(detections_right)
        # if detections_right:
        #     basketball_location_right = detections_right[0]['bbox']
        #     print(f"Basketball at right video detected at: {basketball_location_right}")

        if not windows_created and not GOOGLE_COLAB:
            cv2.namedWindow('Left Video', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Right Video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Left Video', display_width, display_height_left)
            cv2.resizeWindow('Right Video', display_width, display_height_right)
            cv2.moveWindow('Left Video', 0, 0)
            cv2.moveWindow('Right Video', display_width, 0)
            windows_created = True

        # choose active camera based on future detections
        future_frames_list = list(frame_buffer)  # next 10 frames

        left_count = sum(item["left_has_detection"] for item in future_frames_list)
        right_count = sum(item["right_has_detection"] for item in future_frames_list)
        
        if active_camera == "left" and right_count >= left_count:
            active_camera = "right"
        elif active_camera == "right" and left_count >= right_count:
            active_camera = "left"

        item_to_display = frame_buffer.popleft()

        left_frame_to_display = item_to_display["left_frame"].copy()
        right_frame_to_display = item_to_display["right_frame"].copy()

        margin = 10  # just a margin to fit drawn red box correctly
        if active_camera == "left":
            # cv2.putText(left_frame_to_display, "ACTIVE", (20, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.rectangle(left_frame_to_display, (margin, margin),
                          (left_frame_to_display.shape[1] - margin, left_frame_to_display.shape[0] - margin),
                          (0, 0, 255), 16)
        else:
            # cv2.putText(right_frame_to_display, "ACTIVE", (20, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.rectangle(
                right_frame_to_display, 
                (margin, margin),
                (right_frame_to_display.shape[1] - margin, right_frame_to_display.shape[0] - margin),
                (0, 0, 255), 
                16
            )

        if not GOOGLE_COLAB:
            cv2.imshow("Left Video", left_frame_to_display)
            cv2.imshow("Right Video", right_frame_to_display)
        else:
            combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined[0:height_left, 0:width_left] = left_frame_to_display
            combined[0:height_right, width_left:width_left + width_right] = right_frame_to_display
            writer.write(combined)
            
        # Wait for key press (1ms delay)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            print("ESC pressed, exiting")
            break
        elif key == 32:  # Spacebar
            print("Spacebar pressed, pausing. Press spacebar again to resume...")
            while True:
                pause_key = cv2.waitKey(1)
                if pause_key == 32:  # Spacebar again
                    print("Resuming...")
                    break

        # NOTE: ovo je sporo
        # # Jump to next 5th frame
        # frame_count_left += 5
        # frame_count_right += 5
        if GOOGLE_COLAB:
            frame_counter += 1
            if frame_counter >= MAX_FRAMES:
                print("Reached max length")
                break

    # Release resources
    writer.release()

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()