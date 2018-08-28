import pydicom
import glob
import os
import cv2

for file in glob.glob('data/stage_1_train_images/*.dcm'):
  dcm = pydicom.read_file(file)

  patientId = os.path.basename(file)[:-4]

  img = dcm.pixel_array
  res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
  cv2.imwrite('data/stage-1-train-raw/{}.jpeg'.format(patientId), res)
