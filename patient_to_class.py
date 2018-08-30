with open('train_with_boxes.csv', 'r') as file:
  first_line = True
  labels = [0, 0]
  for line in file:
    if first_line:
      first_line = False
      print('patientId,label,male,age,box_x_min_0,box_y_min_0,box_width_0,box_height_0,box_x_min_1,box_y_min_1,box_width_1,box_height_1,box_x_min_2,box_y_min_2,box_width_2,box_height_2,box_x_min_3,box_y_min_3,box_width_3,box_height_3')
      continue

    parts = line.split(',')
    label = int(parts[1].strip())

    if label == 1:
      labels[0] += 1
    else:
      labels[1] += 1

    if labels[0] >= 300 and labels[1] >= 700:
      continue
      print('data/stage-1-train-raw/{}.jpeg,{},{}'.format(parts[0].strip(), label, ','.join(parts[4:]).strip()))
    else:
      # continue
      print('data/stage-1-train-raw/{}.jpeg,{},{}'.format(parts[0].strip(), label, ','.join(parts[4:]).strip()))
