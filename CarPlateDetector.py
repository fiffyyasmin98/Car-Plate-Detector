import tkinter as tk
from tkinter import ttk
from tkinter import TOP, LEFT, W, X, YES , N, S, E, W ,NE, NW, SE, SW
from tkinter import filedialog
from tkinter import Toplevel, Button, RIGHT
from tkinter import Frame, Canvas, CENTER
from PIL import Image, ImageTk
import pygame as pg
import numpy as np
import cv2
import os
import math
import random

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
kNearest = cv2.ml.KNearest_create()
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0
MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
choice = None

class Main(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        # pg.init()
        # pg.mixer.music.load('Still With You.wav')
        # pg.mixer.music.play(-1)
        # pg.mixer.music.set_volume(.1)

        self.filename = ""
        self.original_image = None
        self.original2_image = None
        self.processed_image = None
        self.save_file_type_frame = None
        self.is_image_selected = False
        self.is_canny_state = False
        self.is_prewitt_state = False
        self.is_sobel_state = False
        self.is_detectObj_state = False
        self.is_thinning_state = False
        self.is_detectFeature_state = False
        self.is_size_state = False
        self.is_crop_state = False
        self.merge_frame = None

        self.flip_frame = None
        self.rotate_frame = None
        self.resize_frame = None
        self.translate_frame = None
        self.color_frame = None
        self.adjust_frame = None
        self.filter_frame = None
        self.MergeSplit_frame = None
        self.segment_frame = None
        self.save_as_type_frame = None

        def center(e):
            w = int(self.winfo_width() / 3.5)  # get root width and scale it ( in pixels )
            s = 'CAR PLATE DETECTOR'.rjust(w // 2)
            self.title(s)

        self.bind("<Configure>", center)  # called when window resized
        # self.title("Image Editor")
        self.iconphoto(False, tk.PhotoImage(file='icon.png'))
        # self.configure(bg="blue")
        load = Image.open('bg5.jpg')
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        background_label = tk.Label(self, image=render)
        background_label.image = render
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.editbar1 = EditBar1(master=self)
        self.editbar2 = EditBar2(master=self)
        separator = ttk.Separator(master=self, orient=tk.HORIZONTAL)
        separator1 = ttk.Separator(master=self, orient=tk.HORIZONTAL)
        separator2 = ttk.Separator(master=self, orient=tk.HORIZONTAL)
        self.image_viewer = ImageViewer(master=self)

        separator.pack(fill=tk.X, padx=20, pady=5)
        self.editbar1.pack(pady=5)
        separator1.pack(fill=tk.X, padx=200, pady=5)
        self.editbar2.pack(pady=5)
        separator2.pack(fill=tk.X, padx=20, pady=5)
        self.image_viewer.pack(fill=tk.BOTH, padx=60, pady=20, expand=1)


class EditBar1(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        newicon = tk.PhotoImage(file='new.png').subsample(4,4)
        new2icon = tk.PhotoImage(file='new22.png').subsample(2,2)
        saveicon = tk.PhotoImage(file='save.png').subsample(4,4)
        saveasicon = tk.PhotoImage(file='save as.png').subsample(4,4)
        saveastypeicon = tk.PhotoImage(file='save as type.png').subsample(4,4)
        clearicon = tk.PhotoImage(file='clear.png').subsample(4,4)

        self.new_button = Button(self,  image=newicon,bg='#cce7e8')
        self.new2_button = Button(self, image=new2icon,bg='#cce7e8')
        self.save_button = Button(self, image=saveicon,bg='#cce7e8')
        self.save_as_button = Button(self, image=saveasicon,bg='#cce7e8')
        self.save_as_type_button = Button(self, image=saveastypeicon,bg='#cce7e8')
        self.clear_button = Button(self, image=clearicon,bg='#cce7e8')

        self.new_button.image = newicon
        self.new2_button.image = new2icon
        self.save_button.image = saveicon
        self.save_as_button.image = saveasicon
        self.save_as_type_button.image = saveastypeicon
        self.clear_button.image = clearicon

        self.new_button.bind("<ButtonRelease>", self.new_button_released)
        self.new2_button.bind("<ButtonRelease>", self.new2_button_released)
        self.save_button.bind("<ButtonRelease>", self.save_button_released)
        self.save_as_button.bind("<ButtonRelease>", self.save_as_button_released)
        self.save_as_type_button.bind("<ButtonRelease>", self.save_as_type_button_released)
        self.clear_button.bind("<ButtonRelease>", self.clear_button_released)

        self.new_button.pack(side=LEFT)
        self.new2_button.pack(side=LEFT)
        self.save_button.pack(side=LEFT)
        self.save_as_button.pack(side=LEFT)
        self.save_as_type_button.pack(side=LEFT)
        self.clear_button.pack()

    def new_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.new_button:
            if self.master.is_crop_state:
                self.master.image_viewer.deactivate_crop()
            if self.master.is_canny_state:
                self.master.image_viewer.deactivate_canny()
            if self.master.is_prewitt_state:
                self.master.image_viewer.deactivate_prewitt()
            if self.master.is_sobel_state:
                self.master.image_viewer.deactivate_sobel()
            if self.master.is_thinning_state:
                self.master.image_viewer.deactivate_thinning()
            if self.master.is_detectFeature_state:
                self.master.image_viewer.deactivate_detectFeature()
            if self.master.is_size_state:
                self.master.image_viewer.deactivate_size()
            if self.master.is_detectObj_state:
                self.master.image_viewer.deactivate_detectObj()


            filename = filedialog.askopenfilename()
            image = cv2.imread(filename)

            if image is not None:
                self.master.filename = filename
                self.master.original_image = image.copy()
                self.master.processed_image = image.copy()
                self.master.image_viewer.show_image()
                self.master.is_image_selected = True

    def new2_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.new2_button:
            if self.master.is_crop_state:
                self.master.image_viewer.deactivate_crop()
            if self.master.is_canny_state:
                self.master.image_viewer.deactivate_canny()
            if self.master.is_prewitt_state:
                self.master.image_viewer.deactivate_prewitt()
            if self.master.is_sobel_state:
                self.master.image_viewer.deactivate_sobel()
            if self.master.is_thinning_state:
                self.master.image_viewer.deactivate_thinning()
            if self.master.is_detectFeature_state:
                self.master.image_viewer.deactivate_detectFeature()
            if self.master.is_size_state:
                self.master.image_viewer.deactivate_size()
            if self.master.is_detectObj_state:
                self.master.image_viewer.deactivate_detectObj()

            filename = filedialog.askopenfilename()
            image2 = cv2.imread(filename)

            if image2 is not None:
                self.master.filename = filename
                self.master.original2_image = image2.copy()
                self.master.processed2_image = image2.copy()
                self.master.image_viewer.show_image()
                self.master.is_image_selected = True

    def save_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                save_image = self.master.processed_image
                image_filename = self.master.filename
                cv2.imwrite(image_filename, save_image)

    def save_as_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_as_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                original_file_type = self.master.filename.split('.')[-1]
                filename = filedialog.asksaveasfilename()
                filename = filename + "." + original_file_type

                save_image = self.master.processed_image
                cv2.imwrite(filename, save_image)

                self.master.filename = filename

    def save_as_type_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_as_type_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                self.master.save_as_type_frame = FileTypeFrame(master=self.master)
                self.master.save_as_type_frame.grab_set()

    def clear_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.clear_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                self.master.processed_image = self.master.original_image.copy()
                self.master.image_viewer.show_image()
                self.master.processed2_image = self.master.original2_image.copy()
                self.master.image_viewer.show_image()

class EditBar2(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        detectObjicon = tk.PhotoImage(file='detect object.png').subsample(2,2)
        detectFeatureicon = tk.PhotoImage(file='detect feature.png').subsample(2,2)
        sizeicon = tk.PhotoImage(file='size.png').subsample(2,2)
        thinningicon = tk.PhotoImage(file='thinning.png').subsample(2,2)
        mergeicon = tk.PhotoImage(file='merge.png').subsample(2,2)
        cropicon = tk.PhotoImage(file='crop.png').subsample(2,2)
        cannyicon = tk.PhotoImage(file='canny.png').subsample(2,2)
        prewitticon = tk.PhotoImage(file='prewitt.png').subsample(2,2)
        sobelicon = tk.PhotoImage(file='sobel.png').subsample(2,2)

        self.detectObj_button = Button(self, image=detectObjicon,bg='#cce7e8')
        self.detectFeature_button = Button(self, image=detectFeatureicon,bg='#cce7e8')
        self.size_button = Button(self, image=sizeicon,bg='#cce7e8')
        self.thinning_button = Button(self, image=thinningicon,bg='#cce7e8')
        self.merge_button = Button(self, image=mergeicon,bg='#cce7e8')
        self.crop_button = Button(self, image=cropicon,bg='#cce7e8')
        self.canny_button = Button(self, image=cannyicon,bg='#cce7e8')
        self.prewitt_button = Button(self, image=prewitticon,bg='#cce7e8')
        self.sobel_button = Button(self, image=sobelicon,bg='#cce7e8')

        self.detectObj_button.image = detectObjicon
        self.detectFeature_button.image = detectFeatureicon
        self.size_button.image = sizeicon
        self.thinning_button.image = thinningicon
        self.merge_button.image = mergeicon
        self.crop_button.image = cropicon
        self.canny_button.image = cannyicon
        self.prewitt_button.image = prewitticon
        self.sobel_button.image = sobelicon

        self.detectObj_button.bind("<ButtonRelease>", self.detectObj_button_released)
        self.detectFeature_button.bind("<ButtonRelease>", self.detectFeature_button_released)
        self.size_button.bind("<ButtonRelease>", self.size_button_released)
        self.thinning_button.bind("<ButtonRelease>", self.thinning_button_released)
        self.merge_button.bind("<ButtonRelease>", self.merge_button_released)
        self.crop_button.bind("<ButtonRelease>", self.crop_button_released)
        self.canny_button.bind("<ButtonRelease>", self.canny_button_released)
        self.prewitt_button.bind("<ButtonRelease>", self.prewitt_button_released)
        self.sobel_button.bind("<ButtonRelease>", self.sobel_button_released)

        self.detectObj_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.detectFeature_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.size_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.thinning_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.merge_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.crop_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.canny_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.prewitt_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)
        self.sobel_button.pack(side=LEFT, anchor=W, fill=X, expand=YES)

    def detectObj_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.detectObj_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                else:
                    self.master.image_viewer.activate_detectObj()

    def detectFeature_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.detectFeature_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()
                else:
                    self.master.image_viewer.activate_detectFeature()

    def size_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.size_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                else:
                    self.master.image_viewer.activate_size()

    def thinning_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.thinning_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                else:
                    self.master.image_viewer.activate_thinning()

    def crop_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.crop_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                else:
                    self.master.image_viewer.activate_crop()

    def merge_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.merge_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                self.master.merge_frame = MergeFrame(master=self.master)
                self.master.merge_frame.grab_set()

    def canny_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.canny_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                else:
                    self.master.image_viewer.activate_canny()

    def prewitt_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.prewitt_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                else:
                    self.master.image_viewer.activate_prewitt()

    def sobel_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.sobel_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_canny_state:
                    self.master.image_viewer.deactivate_canny()
                if self.master.is_prewitt_state:
                    self.master.image_viewer.deactivate_prewitt()
                if self.master.is_sobel_state:
                    self.master.image_viewer.deactivate_sobel()
                if self.master.is_thinning_state:
                    self.master.image_viewer.deactivate_thinning()
                if self.master.is_detectFeature_state:
                    self.master.image_viewer.deactivate_detectFeature()
                if self.master.is_size_state:
                    self.master.image_viewer.deactivate_size()
                if self.master.is_detectObj_state:
                    self.master.image_viewer.deactivate_detectObj()

                else:
                    self.master.image_viewer.activate_sobel()



class FileTypeFrame(Toplevel):

    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        self.original_image = self.master.processed_image
        self.filtered_image = None

        self.bmp_button = Button(master=self, text="Bitmaps Type")
        self.jpeg_button = Button(master=self, text="JPEG Type")
        self.tiff_button = Button(master=self, text="TIFF Type")
        self.png_button = Button(master=self, text="PNG Type")
        self.cancel_button = Button(master=self, text="Cancel")

        self.bmp_button.bind("<ButtonRelease>", self.bmp_button_released)
        self.jpeg_button.bind("<ButtonRelease>", self.jpeg_button_released)
        self.tiff_button.bind("<ButtonRelease>", self.tiff_button_released)
        self.png_button.bind("<ButtonRelease>", self.png_button_released)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)

        self.bmp_button.pack()
        self.jpeg_button.pack()
        self.tiff_button.pack()
        self.png_button.pack()
        self.cancel_button.pack(side=RIGHT)

    def bmp_button_released(self, event):
        self.bmp()

    def jpeg_button_released(self, event):
        self.jpeg()

    def tiff_button_released(self, event):
        self.tiff()

    def png_button_released(self, event):
        self.png()

    def cancel_button_released(self, event):
        self.master.image_viewer.show_image()
        self.close()

    def bmp(self):
        type_filename = filedialog.asksaveasfilename()
        type_filename = type_filename + ".bmp"

        save_image = self.master.processed_image
        cv2.imwrite(type_filename, save_image)

        self.master.filename = type_filename

    def jpeg(self):
        type_filename = filedialog.asksaveasfilename()
        type_filename = type_filename + ".jpeg"
        save_image = self.master.processed_image
        cv2.imwrite(type_filename, save_image)

        self.master.filename = type_filename

    def tiff(self):
        type_filename = filedialog.asksaveasfilename()
        type_filename = type_filename + ".tiff"

        save_image = self.master.processed_image
        cv2.imwrite(type_filename, save_image)

        self.master.filename = type_filename

    def png(self):
        type_filename = filedialog.asksaveasfilename()
        type_filename = type_filename + ".png"

        save_image = self.master.processed_image
        cv2.imwrite(type_filename, save_image)

        self.master.filename = type_filename

    def close(self):
        self.destroy()



class MergeFrame(Toplevel):

    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        self.original_image = self.master.processed_image
        self.original2_image = self.master.processed2_image
        self.edited_image = None

        self.mergeH_button = Button(master=self, text="Merge Horizontally")
        self.mergeV_button = Button(master=self, text="Merge Vertically")
        self.cancel_button = Button(master=self, text="Cancel")
        self.apply_button = Button(master=self, text="Apply")

        self.mergeH_button.bind("<ButtonRelease>", self.mergeH_button_released)
        self.mergeV_button.bind("<ButtonRelease>", self.mergeV_button_released)
        self.apply_button.bind("<ButtonRelease>", self.apply_button_released)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)

        self.mergeH_button.pack()
        self.mergeV_button.pack()
        self.apply_button.pack()
        self.cancel_button.pack(side=RIGHT)

    def mergeH_button_released(self, event):
        self.mergeH()
        self.show_image(self.edited_image)

    def mergeV_button_released(self, event):
        self.mergeV()
        self.show_image(self.edited_image)

    def apply_button_released(self, event):
        self.master.processed_image = self.edited_image
        self.close()

    def cancel_button_released(self, event):
        self.master.image_viewer.show_image()
        self.close()

    def show_image(self, img=None):
        self.master.image_viewer.show_image(img=img)

    def mergeH(self, interpolation=cv2.INTER_CUBIC):
        img1 = self.original_image
        img2 = self.original2_image
        img_list = [img1, img2]
        h_min = min(img.shape[0]
                    for img in img_list)

        # image resizing
        im_list_hresize = [
            cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation) for img
            in
            img_list]

        self.edited_image = cv2.hconcat(im_list_hresize)

    def mergeV(self,interpolation=cv2.INTER_CUBIC):
        img1 = self.original_image
        img2 = self.original2_image
        img_list = [img1, img2]

        w_min = min(img.shape[1]
                    for img in img_list)

        # resizing images
        im_list_vresize = [
            cv2.resize(img, (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation=interpolation) for img in
            img_list]

        self.edited_image = cv2.vconcat(im_list_vresize)

    def close(self):
        self.destroy()


class ImageViewer(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master=master, bg="#94D9FF", width=800, height=500)

        self.shown_image = None
        self.x = 0
        self.y = 0
        self.crop_start_x = 0
        self.crop_start_y = 0
        self.crop_end_x = 0
        self.crop_end_y = 0
        self.draw_ids = list()
        self.rectangle_id = 0
        self.ratio = 0
        self.canvas = Canvas(self, bg="#BFE9FF", width=800, height=500)
        self.canvas.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.canvas2 = Canvas(self, bg="#CBFDFF", width=200, height=200)
        self.canvas2.place(relx=0.87, rely=0.8, anchor=CENTER)
        self.canvas3 = Canvas(self, bg="#CBFDFF", width=200, height=200)
        self.canvas3.place(relx=0.13, rely=0.8, anchor=CENTER)

    def show_image(self, img=None):
        self.clear_canvas()

        if img is None:
            image3 = self.master.original_image.copy()
            image = self.master.processed_image.copy()
            image2 = self.master.processed2_image.copy()
        else:
            image3 = self.master.original_image.copy()
            image = img
            image2 = self.master.processed2_image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        ratio = height / width

        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        height2, width2, channels2 = image2.shape
        ratio2 = height2 / width2

        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        height3, width3, channels3 = image3.shape
        ratio3 = height3 / width3

        new_width = width
        new_height = height
        new_width2 = width2
        new_height2 = height2
        new_width3 = width3
        new_height3 = height3

        if height > 600 or width > 800:
            if ratio < 1:
                new_width = 800
                new_height = int(new_width * ratio)
            else:
                new_height = 600
                new_width = int(new_height * (width / height))

        if height2 > 200 or width2 > 200:
            if ratio2 < 1:
                new_width2 = 200
                new_height2 = int(new_width2 * ratio2)
            else:
                new_height2 = 200
                new_width2 = int(new_height2 * (width2 / height2))

        if height3 > 200 or width3 > 200:
            if ratio3 < 1:
                new_width3 = 200
                new_height3 = int(new_width3 * ratio3)
            else:
                new_height3 = 200
                new_width3 = int(new_height3 * (width3 / height3))

        self.shown_image = cv2.resize(image, (new_width, new_height))
        self.shown_image = ImageTk.PhotoImage(Image.fromarray(self.shown_image))
        self.shown2_image = cv2.resize(image2, (new_width2, new_height2))
        self.shown2_image = ImageTk.PhotoImage(Image.fromarray(self.shown2_image))
        self.shown3_image = cv2.resize(image3, (new_width3, new_height3))
        self.shown3_image = ImageTk.PhotoImage(Image.fromarray(self.shown3_image))

        self.ratio = height / new_height
        self.ratio2 = height2 / new_height2
        self.ratio3 = height3 / new_height3

        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width / 2, new_height / 2, anchor=CENTER, image=self.shown_image)
        self.canvas2.config(width=new_width2, height=new_height2)
        self.canvas2.create_image(new_width2 / 2, new_height2 / 2, anchor=CENTER, image=self.shown2_image)
        self.canvas3.config(width=new_width3, height=new_height3)
        self.canvas3.create_image(new_width3 / 2, new_height3 / 2, anchor=CENTER, image=self.shown3_image)

    def activate_crop(self):
        self.canvas.bind("<ButtonPress>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.crop)
        self.canvas.bind("<ButtonRelease>", self.end_crop)

        self.master.is_crop_state = True

    def deactivate_crop(self):
        self.canvas.unbind("<ButtonPress>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease>")

        self.master.is_crop_state = False

    def start_crop(self, event):
        self.crop_start_x = event.x
        self.crop_start_y = event.y

    def crop(self, event):
        if self.rectangle_id:
            self.canvas.delete(self.rectangle_id)

        self.crop_end_x = event.x
        self.crop_end_y = event.y

        self.rectangle_id = self.canvas.create_rectangle(self.crop_start_x, self.crop_start_y,
                                                         self.crop_end_x, self.crop_end_y, width=1)

    def end_crop(self, event):
        if self.crop_start_x <= self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x > self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x <= self.crop_end_x and self.crop_start_y > self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)
        else:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)

        x = slice(start_x, end_x, 1)
        y = slice(start_y, end_y, 1)

        self.master.processed_image = self.master.processed_image[y, x]

        self.show_image()

    def activate_size(self):
        self.detectObj()

        # cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
        # cv2.imshow("imgThresh", licPlate.imgThresh)

        self.drawRedRectangleAroundPlate(self.imgOriginalScene, self.licPlate)

        self.writeSizeOnImage(self.imgOriginalScene, self.licPlate)

        self.master.processed_image = self.imgOriginalScene

        self.show_image()

    def deactivate_size(self):
        pass

    def activate_detectFeature(self):
        self.detectObj()

        # cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
        # cv2.imshow("imgThresh", licPlate.imgThresh)

        self.drawRedRectangleAroundPlate(self.imgOriginalScene, self.licPlate)

        self.writeLicensePlateCharsOnImage(self.imgOriginalScene, self.licPlate)

        self.master.processed_image = self.imgOriginalScene

        self.show_image()

    def deactivate_detectFeature(self):
        pass

    def activate_detectObj(self):
        global choice
        choice = 'Choice 4'
        self.detectObj()

        # cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
        # cv2.imshow("imgThresh", licPlate.imgThresh)

        self.drawRedRectangleAroundPlate(self.imgOriginalScene, self.licPlate)
        self.show_image()

    def deactivate_detectObj(self):
        pass

    def activate_thinning(self):
        retval,imgThresh2=cv2.threshold(self.licPlate.imgThresh,62,255,cv2.THRESH_BINARY_INV)
        self.master.processed_image = imgThresh2
        self.show_image()

    def deactivate_thinning(self):
        pass

    def activate_canny(self):
        global choice
        choice = 'Choice 1'
        self.detectObj()
        self.drawRedRectangleAroundPlate(self.imgOriginalScene, self.licPlate)
        self.show_image()

    def activate_prewitt(self):
        global choice
        choice = 'Choice 2'
        self.detectObj()

        self.drawRedRectangleAroundPlate(self.imgOriginalScene, self.licPlate)
        self.show_image()

    def activate_sobel(self):
        global choice
        choice = 'Choice 3'
        self.detectObj()

        self.drawRedRectangleAroundPlate(self.imgOriginalScene, self.licPlate)
        self.show_image()

    def clear_canvas(self):
        self.canvas.delete("all")

    def detectObj(self):
        blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN()

        if blnKNNTrainingSuccessful == False:
            print("\nerror: KNN traning was not successful\n")
            return

        self.imgOriginalScene = self.master.processed_image

        if self.imgOriginalScene is None:
            print("\nerror: image not read from file \n\n")
            os.system("pause")
            return

        listOfPossiblePlates = detectPlatesInScene(self.imgOriginalScene)

        listOfPossiblePlates = detectCharsInPlates(listOfPossiblePlates)

        if len(listOfPossiblePlates) == 0:
            print("\nno license plates were detected\n")
        else:

            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

            self.licPlate = listOfPossiblePlates[0]

    def drawRedRectangleAroundPlate(self, imgOriginalScene, licPlate):

        p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

        cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

        self.master.processed_image = imgOriginalScene

    def writeLicensePlateCharsOnImage(self, imgOriginalScene, licPlate):

        sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
        plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

        intFontFace = cv2.FONT_HERSHEY_SIMPLEX
        fltFontScale = float(plateHeight) / 50.0
        intFontThickness = int(round(fltFontScale * 1.5))

        textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,intFontThickness)

        ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

        intPlateCenterX = int(intPlateCenterX)
        intPlateCenterY = int(intPlateCenterY)

        ptCenterOfTextAreaX = int(intPlateCenterX)

        if intPlateCenterY < (sceneHeight * 0.75):
            ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
        else:
            ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

        textSizeWidth, textSizeHeight = textSize
        ptLowerLeftTextOriginX = int(
            ptCenterOfTextAreaX - (textSizeWidth / 2))
        ptLowerLeftTextOriginY = int(
            ptCenterOfTextAreaY + (textSizeHeight / 2))

        cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
        # cv2.putText(imgOriginalScene, "{:.1f}in".format(wid), (ptLowerLeftTextOriginX+20, ptLowerLeftTextOriginY+40), intFontFace,
        #             fltFontScale, SCALAR_YELLOW, intFontThickness)
        # cv2.putText(imgOriginalScene, "{:.1f}in".format(ht), ((ptLowerLeftTextOriginX+280), (ptLowerLeftTextOriginY+100)),intFontFace,
        #             fltFontScale, SCALAR_YELLOW, intFontThickness)

    def writeSizeOnImage(self, imgOriginalScene, licPlate):
        ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
        ptCenterOfTextAreaY = 0

        ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
        ptLowerLeftTextOriginY = 0

        sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
        plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

        intFontFace = cv2.FONT_HERSHEY_SIMPLEX
        fltFontScale = float(plateHeight) / 50.0
        intFontThickness = int(round(fltFontScale * 1.5))

        textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,intFontThickness)

        ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

        intPlateCenterX = int(intPlateCenterX)
        intPlateCenterY = int(intPlateCenterY)

        ptCenterOfTextAreaX = int(intPlateCenterX)

        if intPlateCenterY < (sceneHeight * 0.75):
            ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
        else:
            ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

        textSizeWidth, textSizeHeight = textSize
        ptLowerLeftTextOriginX = int(
            ptCenterOfTextAreaX - (textSizeWidth / 2))
        ptLowerLeftTextOriginY = int(
            ptCenterOfTextAreaY + (textSizeHeight / 2))

        pixels_per_metric = 150 / 0.955
        # pixels_per_metric = 150 / (0.955 * 2.54)
        wid = plateWidth / pixels_per_metric
        ht = plateHeight / pixels_per_metric

        # cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
        cv2.putText(imgOriginalScene, "{:.1f}in".format(wid), (intPlateCenterX - 50, intPlateCenterY - 40), intFontFace,
                    fltFontScale, SCALAR_YELLOW, intFontThickness)
        cv2.putText(imgOriginalScene, "{:.1f}in".format(ht), ((intPlateCenterX + 100), (intPlateCenterY + 20)),
                    intFontFace,fltFontScale, SCALAR_YELLOW, intFontThickness)

class PossibleChar:

    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)

class PossiblePlate:

    # constructor #################################################################################
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh

def canny( imgOriginal):
    # imgGrayscale = extractValue(imgOriginal)
    # imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    # height, width = imgGrayscale.shape
    # imgBlurred = np.zeros((height, width, 1), np.uint8)
    # imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (9, 9), 0)
    edged = cv2.Canny(imgBlurred, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    imgThresh = cv2.erode(edged, None, iterations=1)

    return imgGrayscale, imgThresh

def prewitt( imgOriginal):
    # imgGrayscale = extractValue(imgOriginal)
    # imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    # height, width = imgGrayscale.shape
    # imgBlurred = np.zeros((height, width, 1), np.uint8)
    # imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (9, 9), 0)

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(imgBlurred, -1, kernelx)
    img_prewitty = cv2.filter2D(imgBlurred, -1, kernely)
    edged = img_prewittx + img_prewitty
    edged = cv2.dilate(edged, None, iterations=1)
    kernel_sharpening = np.array(
        [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 25, -1, -1], [-1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1]])
    sharpen=cv2.filter2D(edged, -1, kernel_sharpening)
    imgThresh = cv2.erode(sharpen, None, iterations=1)

    return imgGrayscale, imgThresh

def sobel( imgOriginal):
    # imgGrayscale = extractValue(imgOriginal)
    # imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    # height, width = imgGrayscale.shape
    # imgBlurred = np.zeros((height, width, 1), np.uint8)
    # imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (9, 9), 0)
    sobelx = cv2.Sobel(imgBlurred,cv2.CV_8U,1,0,ksize=3)
    sobely = cv2.Sobel(imgBlurred,cv2.CV_8U,0,1,ksize=3)
    edged = sobelx + sobely
    _, img2 = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThresh = img2.copy()
    # edged = cv2.dilate(edged, None, iterations=1)
    # imgThresh = cv2.erode(edged, None, iterations=1)

    return imgGrayscale, imgThresh

def getThreshGray (imgOriginal):
    global choice
    if choice == 'Choice 1':
        imgThresh, imgGrayscale = canny(imgOriginal)
    elif choice == 'Choice 2':
        imgThresh, imgGrayscale = prewitt(imgOriginal)
    elif choice == 'Choice 3':
        imgThresh, imgGrayscale = sobel(imgOriginal)
    elif choice == 'Choice 4':
        imgThresh, imgGrayscale = preprocess(imgOriginal)
    else:
        imgThresh, imgGrayscale=preprocess(imgOriginal)

    return imgThresh, imgGrayscale

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue

def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

def loadKNNDataAndTrainKNN():
    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return False

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest.setDefaultK(1)

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    return True

def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates

    for possiblePlate in listOfPossiblePlates:
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = getThreshGray(possiblePlate.imgPlate)

        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY |cv2.THRESH_OTSU)

        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if (len(listOfListsOfMatchingCharsInPlate) == 0):

            possiblePlate.strChars = ""
            continue

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])

        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i

        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

    return listOfPossiblePlates

def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []
    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possibleChar = PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)

    return listOfPossibleChars

def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
        listOfMatchingChars.append(possibleChar)

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue
        listOfListsOfMatchingChars.append(listOfMatchingChars)

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

        break

    return listOfListsOfMatchingChars

def findListOfMatchingChars(possibleChar, listOfChars):
    listOfMatchingChars = []

    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:
            continue

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)


        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)

    return listOfMatchingChars

def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)

    return fltAngleInDeg

def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)
    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)
                    else:
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)

    return listOfMatchingCharsWithInnerCharRemoved

def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, SCALAR_GREEN, 2)

                # crop char out of threshold image
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))

        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)

        strCurrentChar = str(chr(int(npaResults[0][0])))

        strChars = strChars + strCurrentChar

    return strChars

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    imgGrayscaleScene, imgThreshScene = getThreshGray(imgOriginalScene)

    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    listOfListsOfMatchingCharsInScene = findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)

        if possiblePlate.imgPlate is not None:
            listOfPossiblePlates.append(possiblePlate)
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")

    return listOfPossiblePlates

def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):

        possibleChar = PossibleChar(contours[i])

        if checkIfPossibleChar(possibleChar):
            intCountOfPossibleChars = intCountOfPossibleChars + 1
            listOfPossibleChars.append(possibleChar)

    return listOfPossibleChars
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate()

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped

    return possiblePlate


root = Main()
root.mainloop()