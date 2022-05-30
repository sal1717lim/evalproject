import os
import sys
import moviepy.video.io.ImageSequenceClip
image_folder=sys.argv[1]
fps=30

image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)]
print("start video making")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=5)
print("save")
clip.write_videofile(sys.argv[1]+'.mp4')