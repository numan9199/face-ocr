import face_recognition
import cv2
import os
import glob
import numpy as np
import pickle

class EncodeFace:
    def __init__(self):
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        known_face_encodings = []
        known_face_names = []
       
        fist_loop = True
        for img_path in images_path:
            img = face_recognition.load_image_file(img_path)

            
            
            basename = os.path.basename(img_path)
            
            (filename, ext) = os.path.splitext(basename)
         
            img_loc= face_recognition.face_locations(img)
            img_encoding = face_recognition.face_encodings(img,img_loc)[0]

            
            known_face_encodings.append(img_encoding)
            known_face_names.append(filename)
            if fist_loop :
                with open('know_face_names.p','wb') as f:
                    pickle.dump((filename), f) 
                with open('know_face_encodes.p','wb') as f:
                    pickle.dump((img_encoding), f)
                fist_loop = False
            else:
                with open('know_face_names.p','ab') as f:
                    pickle.dump((filename), f) 
                with open('know_face_encodes.p','ab') as f:
                    pickle.dump((img_encoding), f)
        print("Encoding images loaded")
         
        
        


        

   