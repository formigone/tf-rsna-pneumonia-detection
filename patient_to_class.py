with open('stage_1_train_labels.csv', 'r') as file:
  first_line = True
  ids = {}
  labels = [0, 0]
  for line in file:
    if first_line:
      first_line = False
      print('patientId,Target')
      continue

    parts = line.split(',')
    label = int(parts[-1].strip())
    if parts[0] in ids:
      continue

    if label == 1:
      labels[0] += 1
    else:
      labels[1] += 1
    ids[parts[0]] = True

    if labels[0] >= 300 and labels[1] >= 700:
      continue
      print('data/stage-1-train-raw/{}.jpeg,{}'.format(parts[0].strip(), label))
    else:
      # continue
      print('data/stage-1-train-raw/{}.jpeg,{}'.format(parts[0].strip(), label))
