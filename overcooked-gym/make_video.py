import cv2


def make_video(episode_directory, videofilename, fps=5):

    img_array = []
    step = 0
    while True:
        filename = f"{episode_directory}/step_{step}.png"
        img = cv2.imread(filename)
        if img is not None:
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
            step += 1
        else:
            break

    out = cv2.VideoWriter(videofilename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':
    directory = "resources/episodes/episode_3_frames"
    filename = "resources/episodes/episode_3.mp4"

    print("Making video...")
    make_video(directory, filename)
    print("Done!")
