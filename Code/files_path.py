import os

for i in range(2000, 2301):
    #for j in range(6):
    with open("/home/ivan/Desktop/pic_paths_test.txt", "a+") as f:
        if os.path.isfile("/home/ivan/linux mint/Frames/frame" + str(i) + ".txt"):
            f.write("/content/drive/My Drive/test/frame" + str(i) + ".png\n")
