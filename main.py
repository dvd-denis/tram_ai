import numpy as np
import argparse
import imutils
import cv2
import math
import os
import tkinter
import customtkinter

from time import sleep
from imutils.video import FileVideoStream
from imutils.video import FPS
from PIL import Image
from tkintermapview import TkinterMapView

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]

    dot_prod = dot(vA, vB)

    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5

    cos_ = dot_prod/magA/magB
    angle = math.acos(dot_prod/magB/magA)
    ang_deg = math.degrees(angle)%360
    
    if ang_deg-180>=0:
        return 360 - ang_deg
    else: 
        return ang_deg

def on_closing():
	global app_started
	app_started = False
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.destroyAllWindows()
	vs.stop()
	exit()

customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.wm_title("AI")
app.geometry("1000x400")
app.protocol("WM_DELETE_WINDOW", on_closing)
app.bind('<Escape>', lambda e: on_closing())

app.grid_columnconfigure(0, weight=0)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

frame_left = customtkinter.CTkFrame(master=app, height=400, width=600, corner_radius=0, fg_color=None)
frame_left.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

frame_right = customtkinter.CTkFrame(master=app, height=400, width=400, corner_radius=0)
frame_right.grid(row=0, column=1, rowspan=1, pady=0, padx=0, sticky="nsew")

label_ct = customtkinter.CTkLabel(frame_left, text="")
label_ct.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

map_widget = TkinterMapView(frame_right, height=400, width=400, corner_radius=0)
map_widget.grid(row=1, rowspan=1, column=0, columnspan=3, sticky="nswe", padx=(0, 0), pady=(0, 0))
map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
map_widget.set_address("Omsk")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
	
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


print("[INFO] loading model...")

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)	

print("[INFO] starting video stream...")
vs = FileVideoStream("./video.mp4").start()

sleep(2.0)
fps = FPS().start()

app_started = True
def main():
	while app_started:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		net.setInput(blob)
		detections = net.forward()

		stop_check = False

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > 0.2:
				idx = int(detections[0, 0, i, 1])
				if CLASSES[idx] == "train":
					continue

				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				width = endX - startX
				height = endY - startY
				
				posX = startX + width//2
				posY = startY

				lenght = (((200-posX)**2 + (200-posY)) ** 0.5) + (width*height//2)
				cv2.line(frame, (200, 200), (posX, posY), (0,255,0), 3)


				line1 = ((200,200), (200,100))
				line2 = ((200,200), (posX, posY))
				ang1 = ang(line1, line2)

				if ang1 <= 50 and lenght>=2000:
					stop_check = True

				label = "{}: {}".format(int(lenght), int(ang1))

				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
		if stop_check == True:
			os.system('cls')
			print("[AI] stop")
		else:
			os.system('cls')
			print("[AI] continue")
		
		im = Image.fromarray(frame)
		my_image = customtkinter.CTkImage(light_image=im, dark_image=im, size=(600, 400))
		label_ct.configure(image=my_image)

		fps.update()
		map_widget.update()
		
if __name__ == "__main__":
	app.after(100, main)
	app.mainloop()