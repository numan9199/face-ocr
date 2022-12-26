import pickle
from typing import List
from fastapi import FastAPI, File, Form, UploadFile
from starlette.middleware.cors import CORSMiddleware
import io
import face_recognition
import numpy as np
from fastapi.encoders import jsonable_encoder
from PIL import Image, ImageDraw
import cv2
from Encode_face import EncodeFace

#encode available image on start server
EncodeFace().load_encoding_images("./images")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.post("/api/Identify")
async def faces_recognition(image_upload: UploadFile = File(...)):
    data = await image_upload.read()
    known_face_names =[]
    known_face_encodings=[]
    
    image = face_recognition.load_image_file(io.BytesIO(data))
    #img = Image.open(io.BytesIO(data))
    #draw = ImageDraw.Draw(img)

    

    with open('know_face_names.p','rb') as f:
        while 1:
            try:
               known_face_names.append(pickle.load(f))
            except EOFError:
                break
    with open('know_face_encodes.p','rb') as f:
        while 1:
            try:
               known_face_encodings.append(pickle.load(f))
            except EOFError:
                break
    #print(known_face_names)

    # Detect face(s) and encode them
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    
    face_names = []
    face_loc=[]

    # Recognize face(s)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #print(face_distances)
        best_match_index = np.argmin(face_distances)
        #print(best_match_index)
        if matches[best_match_index]:   
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        #top, right, bottom, left = face_location
        #draw.rectangle([left, top, right, bottom],width = 4)
        #draw.text((left, top), name)
        face_names.append(name)
        face_loc.append(face_location)
    #img.show()
    return {"Face name ": face_names,"Face location ": face_loc}



@app.post("/api/AddImg")
async def faces_recognition(image_upload: UploadFile = File(...),name :str =Form()):
    data = await image_upload.read()
    img = Image.open(io.BytesIO(data))
    img.save("./images/{}.png".format(name))
    image = face_recognition.load_image_file(io.BytesIO(data))
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)[0]
    
    with open('know_face_names.p','ab') as f:
        pickle.dump((name), f)
    with open('know_face_encodes.p','ab') as f:
        pickle.dump((face_encodings), f)

    return {"message" : "add success"}



@app.post("/api/AddMultiImg")
async def create_upload_files(files: List[UploadFile],name :str=Form()):
    for data in files:
        data = await data.read()
        image = face_recognition.load_image_file(io.BytesIO(data))
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)[0]
        with open('know_face_names.p','ab') as f:
            pickle.dump((name), f)
        with open('know_face_encodes.p','ab') as f:
            pickle.dump((face_encodings), f)
        
    return {"message":"add success"}





