from __future__ import division

import pydicom
import glob
import pandas as pd

def from_raw_files():
  rows = []

  for file in glob.glob('data/stage_1_train_images/*.dcm'):
    dcm = pydicom.read_file(file)

    # [height, width, id, age, gender] = [dcm.Rows, dcm.Columns, dcm.PatientID, dcm.PatientAge, dcm.PatientSex]
    rows.append([dcm.Rows, dcm.Columns, dcm.PatientID, int(dcm.PatientAge), dcm.PatientSex])
    if len(rows) > 1000:
      break

  rows = pd.DataFrame(rows, columns=['height', 'width', 'patientId', 'age', 'gender'])
  print(rows)
    # img = dcm.pixel_array
    # res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('data/stage-1-train-raw/{}.jpeg'.format(patientId), res)

def from_csv(path):
  with open(path, 'r') as lines:
    ids = {}
    for index, line in enumerate(lines):
      if index == 0:
        continue
      parts = line.split(',')
      patientId = parts[0]
      if patientId not in ids:
        dcm = pydicom.read_file('data/stage_1_train_images/{}.dcm'.format(patientId))
        ids[patientId] = {
          'patientId': patientId,
          'label': int(parts[-1]),
          'img_width': dcm.Columns,
          'img_height': dcm.Rows,
          'male': int(dcm.PatientSex.upper() == 'M'),
          'age': int(dcm.PatientAge),
          'box_x_min_0': 0,
          'box_y_min_0': 0,
          'box_width_0': 0,
          'box_height_0': 0,
          'box_x_min_1': 0,
          'box_y_min_1': 0,
          'box_width_1': 0,
          'box_height_1': 0,
          'box_x_min_2': 0,
          'box_y_min_2': 0,
          'box_width_2': 0,
          'box_height_2': 0,
          'box_x_min_3': 0,
          'box_y_min_3': 0,
          'box_width_3': 0,
          'box_height_3': 0,
        }

      if parts[1]:
        min_box = 0
        for i in range(4):
          if ids[patientId]['box_width_{}'.format(i)] == 0:
            min_box = i
            break
        ids[patientId]['box_x_min_{}'.format(min_box)] = float(parts[1]) / ids[patientId]['img_width']
        ids[patientId]['box_y_min_{}'.format(min_box)] = float(parts[2]) / ids[patientId]['img_height']
        ids[patientId]['box_width_{}'.format(min_box)] = float(parts[3]) / ids[patientId]['img_width']
        ids[patientId]['box_height_{}'.format(min_box)] = float(parts[4]) / ids[patientId]['img_height']

  ids = pd.DataFrame(ids.values(), columns=['patientId', 'label', 'img_width', 'img_height', 'male', 'age', 
          'box_x_min_0',
          'box_y_min_0',
          'box_width_0',
          'box_height_0',
          'box_x_min_1',
          'box_y_min_1',
          'box_width_1',
          'box_height_1',
          'box_x_min_2',
          'box_y_min_2',
          'box_width_2',
          'box_height_2',
          'box_x_min_3',
          'box_y_min_3',
          'box_width_3',
          'box_height_3'])
  ids.to_csv('train_with_boxes.csv', index=False)
  ids.to_json('train_with_boxes.json')
  print(ids)
  exit(0)

from_csv('stage_1_train_labels.csv')
