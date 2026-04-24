from find_files import read_offset_file, select_files
from process_videos import process_videos




if __name__ == "__main__":
    # Select files using file dialogs
    # video_left, video_right, offset_left, offset_right = select_files()
    # offset_left = read_offset_file(offset_left)
    # offset_right = read_offset_file(offset_right)
    # offset_common = 0

    # # short vid
    # video_left = r"C:\Users\kiko\Desktop\LIPIK završni ispit\videos\sl_CED4.mp4"
    # video_right = r"C:\Users\kiko\Desktop\LIPIK završni ispit\videos\sr_CED4.mp4"
    # offset_left = 0
    # offset_right = 0
    # offset_common = 0

    # long vid
    video_left = r"C:\Users\kiko\Desktop\LIPIK završni ispit\videos\ll_CED4.mp4"
    video_right = r"C:\Users\kiko\Desktop\LIPIK završni ispit\videos\lr_CED4.mp4"
    offset_left = 375.00
    offset_right = 4.80
    offset_common = 8  # za pokazati: 7, 56, 86, 108, 122, 132, 213, 243, 257, 

    print(video_left, video_right, offset_left, offset_right, sep='\n')

    process_videos(
        video_left, video_right, offset_left + offset_common, offset_right + offset_common, 
        # "annotated_video.mp4",
    )