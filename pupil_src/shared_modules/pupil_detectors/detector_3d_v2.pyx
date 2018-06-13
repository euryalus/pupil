'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

# cython: profile=False
import cv2
import numpy as np
from coarse_pupil cimport center_surround
from methods import Roi, normalize
from plugin import Plugin
from pyglui import ui
import glfw
from gl_utils import  adjust_gl_view, clear_gl_screen, basic_gl_setup, make_coord_system_norm_based, make_coord_system_pixel_based
from pyglui.cygl.utils import draw_gl_texture
import math

from visualizer_3d import Eye_Visualizer
from collections import namedtuple

from detector cimport *
from detector_utils cimport *

from cython.operator cimport dereference as deref

cdef class Detector_3D_v2:

    cdef Detector2D * detector2DPtr
    cdef EyeModel_v2 * detector3DPtr
    cdef shared_ptr[Detector2DResult] cpp2DResultPtr
    cdef dict detectProperties2D, detectProperties3D
    cdef object debugVisualizer3D
    cdef object pyResult3D
    cdef readonly object g_pool
    cdef readonly basestring uniqueness
    cdef public object menu
    cdef public object menu_icon
    cdef readonly basestring icon_chr
    cdef readonly basestring icon_font

    def __cinit__(self, g_pool = None, settings = None):
        self.detector2DPtr = new Detector2D()
        focal_length = 620.
        cdef Vector3 camera_center
        camera_center[0] = 0.0
        camera_center[1] = 0.0
        camera_center[2] = 0.0
        self.detector3DPtr = new EyeModel_v2(0.0, focal_length, camera_center)

    def __init__(self, g_pool=None, settings=None):
        self.debugVisualizer3D = Eye_Visualizer(g_pool, self.detector3DPtr.getFocalLength())
        self.g_pool = g_pool
        self.uniqueness = 'unique'
        self.icon_font = 'pupil_icons'
        self.icon_chr = chr(0xec19)

        self.detectProperties2D = settings['2D_Settings'] if settings else {}
        self.detectProperties3D = settings['3D_Settings'] if settings else {}

        if not self.detectProperties2D:
            self.detectProperties2D["coarse_detection"] = True
            self.detectProperties2D["coarse_filter_min"] = 128
            self.detectProperties2D["coarse_filter_max"] = 280
            self.detectProperties2D["intensity_range"] = 23
            self.detectProperties2D["blur_size"] = 3
            self.detectProperties2D["canny_treshold"] = 200
            self.detectProperties2D["canny_ration"] = 2
            self.detectProperties2D["canny_aperture"] = 5
            self.detectProperties2D["pupil_size_max"] = 400
            self.detectProperties2D["pupil_size_min"] = 10
            self.detectProperties2D["strong_perimeter_ratio_range_min"] = 0.8
            self.detectProperties2D["strong_perimeter_ratio_range_max"] = 1.1
            self.detectProperties2D["strong_area_ratio_range_min"] = 0.6
            self.detectProperties2D["strong_area_ratio_range_max"] = 1.1
            self.detectProperties2D["contour_size_min"] = 5
            self.detectProperties2D["ellipse_roundness_ratio"] = 0.1
            self.detectProperties2D["initial_ellipse_fit_treshhold"] = 1.8
            self.detectProperties2D["final_perimeter_ratio_range_min"] = 0.6
            self.detectProperties2D["final_perimeter_ratio_range_max"] = 1.2
            self.detectProperties2D["ellipse_true_support_min_dist"] = 2.5
            self.detectProperties2D["support_pixel_ratio_exponent"] = 2.0

        if not self.detectProperties3D:
            self.detectProperties3D["model_sensitivity"] = 0.997

    @property
    def pretty_class_name(self):
        return 'Pupil Detector 3D'

    # Getters and setters for general 2D and 3D settings
    def get_settings(self):
        return {'2D_Settings': self.detectProperties2D , '3D_Settings': self.detectProperties3D}

    def detectProperties2D_setter(self, dict_):
        self.detectProperties2D = dict_

    def detectProperties3D_setter(self, dict_):
        self.detectProperties3D = dict_

    def on_resolution_change(self, old_size, new_size):
        self.detectProperties2D["pupil_size_max"] *= new_size[0] / old_size[0]
        self.detectProperties2D["pupil_size_min"] *= new_size[0] / old_size[0]

    def get_sphere(self):
        cdef Sphere[double] sphere_cpp = self.detector3DPtr.getSphere()
        sphere = {'center': np.array([sphere_cpp.center[0],sphere_cpp.center[1],sphere_cpp.center[2]]),
                  'radius': sphere_cpp.radius}
        return sphere

    def get_initial_sphere(self):
        cdef Sphere[double] sphere_cpp = self.detector3DPtr.getInitialSphere()
        sphere = {'center': np.array([sphere_cpp.center[0],sphere_cpp.center[1],sphere_cpp.center[2]]),
                  'radius': sphere_cpp.radius}
        return sphere

    def get_number_of_supporting_pupils(self):
        return self.detector3DPtr.getSupportingPupilsSize()

    # Relaying observations, performing 2D and 3D detection, etc.
    def detect2D(self, frame, user_roi, visualize):

        image_width = frame.width
        image_height = frame.height

        cdef unsigned char[:,::1] img = frame.gray
        cdef Mat cv_image = Mat(image_height, image_width, CV_8UC1, <void *> &img[0,0] )

        cdef unsigned char[:,:,:] img_color
        cdef Mat cv_image_color
        cdef Mat debug_image

        if visualize:
            img_color = frame.img
            cv_image_color = Mat(image_height, image_width, CV_8UC3, <void *> &img_color[0,0,0] )

        roi = Roi((0,0))
        roi.set(user_roi.get())
        roi_x = roi.get()[0]
        roi_y = roi.get()[1]
        roi_width  = roi.get()[2] - roi.get()[0]
        roi_height  = roi.get()[3] - roi.get()[1]
        cdef int[:,::1] integral

        if self.detectProperties2D['coarse_detection'] and roi_width*roi_height>320*240:
            scale = 2 # Half the integral imag to boost up integral.
            user_roi_image = frame.gray[user_roi.view]
            integral = cv2.integral(user_roi_image[::scale,::scale])
            coarse_filter_max = self.detectProperties2D['coarse_filter_max']
            coarse_filter_min = self.detectProperties2D['coarse_filter_min']
            bounding_box, good_ones, bad_ones = center_surround(integral, coarse_filter_min/scale, coarse_filter_max/scale)

            if visualize:
                # Draw the candidates.
                for v  in good_ones:
                    p_x,p_y,w,response = v
                    x = p_x * scale + roi_x
                    y = p_y * scale + roi_y
                    width = w*scale
                    cv2.rectangle(frame.img, (x,y), (x+width,y+width), (255,255,0))

            x1 , y1 , x2, y2 = bounding_box
            width = x2 - x1
            height = y2 - y1
            roi_x = x1 * scale + roi_x
            roi_y = y1 * scale + roi_y
            roi_width = width*scale
            roi_height = height*scale
            roi.set((roi_x, roi_y, roi_x+roi_width, roi_y+roi_height))

        # All coordinates in the result are relative to the current ROI
        self.cpp2DResultPtr = self.detector2DPtr.detect(self.detectProperties2D, cv_image, cv_image_color, debug_image, Rect_[int](roi_x,roi_y,roi_width,roi_height), visualize , False ) #we don't use debug image in 3d model
        deref(self.cpp2DResultPtr).timestamp = frame.timestamp # Timestamp is not set elsewhere but is needed in EyeModel

        py_result = convertTo2DPythonResult(deref(self.cpp2DResultPtr), frame, roi) # Todo: Not needed in final version, here only for debugging

        return py_result

    def add_observation(self, frame, user_roi, visualize, prepare_observation=True):

        # 2D model part
        self.detect2D(frame, user_roi, visualize) # Shared pointer to result is stored in self.Result2D_ptr
        deref(self.cpp2DResultPtr).timestamp = frame.timestamp # The timestamp is not set elsewhere but it is needed in detector3D

        # 3D model part
        N = self.detector3DPtr.addObservation(self.cpp2DResultPtr, prepare_observation)

        return N

    def optimize_model(self):

        cdef Sphere[double] sphere_cpp = self.detector3DPtr.optimizeModel()
        sphere = {'center': np.array([sphere_cpp.center[0], sphere_cpp.center[1], sphere_cpp.center[2]]),
                  'radius': sphere_cpp.radius}
        return sphere

#    def detect3D(self, frame, user_roi, visualize):
#
#        # 2D model part
#        self.detect2D(frame, user_roi, visualize) # Shared pointer to result is stored in self.Result2D_ptr
#        deref(self.cpp2DResultPtr).timestamp = frame.timestamp # The timestamp is not set elsewhere but it is needed in detector3D
#
#        # 3D model part
#        try:
#            debugDetector = self.debugVisualizer3D.window   # This fails when called outside of Pupil Capture
#        except:
#            debugDetector = True
#
#        cdef Detector3DResult cpp3DResult  = self.detector3DPtr.predictAndUpdate(self.cpp2DResultPtr, self.detectProperties3D, debugDetector)
#        pyResult = convertTo3DPythonResult(cpp3DResult, frame)
#
#        if debugDetector:
#           self.pyResult3D = prepareForVisualization3D(cpp3DResult)
#
#        return pyResult

    # Resetting and deallocation
    def __dealloc__(self):
      del self.detector2DPtr
      del self.detector3DPtr

    # UI and vizualisation
    def init_ui(self):
        Plugin.add_menu(self)
        self.menu.label = self.pretty_class_name
        info = ui.Info_Text("Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "\
                           +"Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "\
                           +"Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds.")
        self.menu.append(info)
        self.menu.append(ui.Slider('intensity_range',self.detectProperties2D,label='Pupil intensity range',min=0,max=60,step=1))
        self.menu.append(ui.Slider('pupil_size_min',self.detectProperties2D,label='Pupil min',min=1,max=250,step=1))
        self.menu.append(ui.Slider('pupil_size_max',self.detectProperties2D,label='Pupil max',min=50,max=400,step=1))
        info_3d = ui.Info_Text("Open the debug window to see a visualization of the 3D pupil detection.")
        self.menu.append(info_3d)
        self.menu.append(ui.Button('Reset 3D model', self.reset_3D_Model))
        self.menu.append(ui.Button('Open debug window',self.toggle_window))
        self.menu.append(ui.Slider('model_sensitivity',self.detectProperties3D,label='Model sensitivity',min=0.990,max=1.0,step=0.0001))
        self.menu[-1].display_format = '%0.4f'

    def deinit_ui(self):
        Plugin.remove_menu(self)

    def toggle_window(self):
        if not self.debugVisualizer3D.window:
            self.debugVisualizer3D.open_window()
        else:
            self.debugVisualizer3D.close_window()

    def cleanup(self):
        self.debugVisualizer3D.close_window() # If we change detectors, be sure debug window is also closed.

    def visualize(self):
        if self.debugVisualizer3D.window:
            self.debugVisualizer3D.update_window(self.g_pool,self.pyResult3D)

