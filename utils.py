# Google Drive upload/doanload helpers
def drive_upload_file(full_filename):
  filename = os.path.basename(full_filename)
  file_to_upload = drive.CreateFile({'title': filename})
  file_to_upload.SetContentFile(full_filename)
  file_to_upload.Upload()
  
def drive_download_file(file_id, local_path='./'):
  # A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
  downloaded = drive.CreateFile({'id': file_id})
  downloaded.FetchMetadata()
  fn = downloaded.metadata.get('originalFilename')
  full_fn = os.path.join(local_path, fn)
  downloaded.GetContentFile(full_fn)
  return full_fn

# Dataset reading helpers
def filter_df(df, min_images_count=0):
  df = df.sort_values('images', ascending=False)
  return df[df.images >= min_images_count]
  
# Undo ImageNet preprocessing to show images from batch generator
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # RGB
de_preproc = lambda x: np.clip((x[..., ::-1]+vgg_mean)/255., 0, 1)

# Visualization helpers

# equal probabilities
equal_apriori = 0.5

# number of thresholds
num_thresholds = 100
# generate a list of  n thresholds between 0.0 and 1.0
thresholds = [i/num_thresholds for i in range(num_thresholds)]

def plot_scores(imposter, genuine):
  """ Plot the scores of the genuine and imposters """

  # Draws a histogram to show score frequencies with values.
  plt.hist(imposter, facecolor='g', alpha=0.50, label='Imposter')
  plt.hist(genuine, facecolor='y', alpha=0.50, label='Genuine')

  # Adding labels and titles to the plot
  plt.xlabel('Score')
  plt.ylabel('Frequency')
  plt.title('Score Distribution')
  plt.grid(True)

  # draw the key/legend
  plot_legends()

  # show the plot
  show_plot()

def calculate_cost(imposter, genuine):
  """ For both users, calculates a confusion matrix and then calculates cost per threshold """

  # generate n number of thresholds

  far = []
  frr = []
  cost = []

  # for each threshold, calculate confusion matrix.
  for t in thresholds:

    FP = 0
    FN = 0
    TP = 0
    TN = 0

    # go through imposters
    for score in imposter:

      if score >= t:
        # imposter passes as a genuine user
        FP += 1
      else:
        # imposter correctly rejected
        TN += 1

    for score in genuine:
      if score >= t:
        # genuine user correctly identified
        TP += 1
      else:
        # genuine user incorrectly rejected
        FN += 1

    far_current = float(FP) / float(len(imposter))
    frr_current = float(FN) / float(len(genuine))

    # calculate our false accept rate(FAR) and add to list
    far.append(far_current)

    # calculate our false reject rate(FRR) and add to list
    frr.append(frr_current)

  return far, frr

def plot_DET_with_EER(far, frr, far_optimum, frr_optimum):
  """ Plots a DET curve with the most suitable operating point based on threshold values"""

  # Plot the DET curve based on the FAR and FRR values
  plt.plot(far, frr, linestyle="--", linewidth=4, label="DET Curve")

  # Plot the optimum point on the DET Curve
  plt.plot(far_optimum,frr_optimum, "ro", label="Suitable Operating Point")

  # Draw the default DET Curve from 1-1
  plt.plot([1.0,0.0], [0.0,1.0],"k--")

  # Draws the key/legend
  plot_legends()

  # Displays plots
  show_plot()
  
def plot_FAR_vs_FRR(far, frr):
  # Plot FAR and FRR
  plt.plot(far, 'r-', label='FAR curve')
  plt.plot(frr, 'g-', label='FRR curve')
  
  # Draws the key/legend
  plot_legends()

  # Displays plots
  show_plot()
  
def find_EER(far, frr):
  """ Returns the most optimal FAR and FRR values """

  # The lower the equal error rate value,
  # the higher the accuracy of the biometric system.

  t = []
  far_optimum = 0
  frr_optimum = 0

  # go through each value for FAR and FRR, calculate
  for i in range(num_thresholds):
    t.append(far[i] + frr[i])

  # smallest value is most accurate
  smallest = min(t)

  for i in range(num_thresholds):
    if smallest == far[i] + frr[i]:

      # Found EER
      far_optimum = far[i]
      frr_optimum = frr[i]
      threshold_optimum = thresholds[i]
      break

  return far_optimum, frr_optimum, threshold_optimum

def plot_legends():
  legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
  legend.get_frame().set_facecolor('#ffffff')

def show_plot():
  plt.show()
  
  
# Import metrics
from scipy.spatial import distance
from sklearn.metrics import accuracy_score

# Import zoom function
from keras_preprocessing.image import random_zoom

# Images root dir
root_dir = 'input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled'

def extract_template(img_fn, img_target_size, template_extrator):
  """
  Read image and extract template from it
  """
  # read image and resize to match model's input
  img = pil_to_array(Image.open(img_fn).resize(img_target_size))
  
  # apply the same preprocessing as in the training phase
  img = random_zoom(img, (0.5, 0.5), 0, 1, 2)
  img = preprocess_input(img)
  img = np.expand_dims(img, axis=0)
  
  # extract embedding template
  template = template_extrator.predict(img)
  template = np.squeeze(template)
  return template.tolist()

def name_to_image_path(root_dir, name, image_num):
  return f'{root_dir}/{name}/{name}_{image_num:04d}.jpg'

def evaluate_model(model, dev_pairs_file='./input/lfw-dataset/course-pairsDevTest.csv', test_pairs_file='./input/lfw-dataset/course-pairs.csv'):
  # Cut our model at the 'embedding' layer level and convert it to template extractor
  embedding_out = model.get_layer('face_embedding').output
  template_extrator = Model(inputs=[input_layer], outputs=[embedding_out])

  # Read pairs matched and mismatched for dev dataset
  print('Preparing stats from dev set')
  test_pairs = pd.read_csv(dev_pairs_file)
  test_pairs['img_fn1'] = test_pairs.apply(lambda row: name_to_image_path(root_dir, row['name1'], row['imagenum1']), axis=1)
  test_pairs['img_fn2'] = test_pairs.apply(lambda row: name_to_image_path(root_dir, row['name2'], row['imagenum2']), axis=1)
  test_pairs['template1'] = test_pairs.progress_apply(lambda row: extract_template(row['img_fn1'], target_size, template_extrator), axis=1)
  test_pairs['template2'] = test_pairs.progress_apply(lambda row: extract_template(row['img_fn2'], target_size, template_extrator), axis=1)
  test_pairs['cos_distance'] = test_pairs.apply(lambda row: distance.cosine(row['template1'], row['template2']), axis=1)
  
  match_scores = test_pairs[test_pairs.match_pair==0]['cos_distance']
  mismatch_scores = test_pairs[test_pairs.match_pair==1]['cos_distance']

  # Plot model's stats
  genuine = match_scores.values
  imposter = mismatch_scores.values

  far, frr = calculate_cost(imposter, genuine)
  far_optimum, frr_optimum, err_threshold = find_EER(far, frr)

  plot_scores(imposter, genuine)
  plot_DET_with_EER(far, frr, far_optimum, frr_optimum)
  plot_FAR_vs_FRR(far, frr)
  print(f'EER at threshold: {err_threshold}')
  
  # Now let's calculate accuracy for test set
  print('Preparing stats from test set')
  test_pairs = pd.read_csv(test_pairs_file)
  test_pairs['img_fn1'] = test_pairs.apply(lambda row: name_to_image_path(root_dir, row['name1'], row['imagenum1']), axis=1)
  test_pairs['img_fn2'] = test_pairs.apply(lambda row: name_to_image_path(root_dir, row['name2'], row['imagenum2']), axis=1)
  test_pairs['template1'] = test_pairs.progress_apply(lambda row: extract_template(row['img_fn1'], target_size, template_extrator), axis=1)
  test_pairs['template2'] = test_pairs.progress_apply(lambda row: extract_template(row['img_fn2'], target_size, template_extrator), axis=1)
  test_pairs['cos_distance'] = test_pairs.apply(lambda row: distance.cosine(row['template1'], row['template2']), axis=1)
  test_pairs['pred_match_pair'] = test_pairs['cos_distance'] < err_threshold
  test_pairs['pred_match_pair'] = test_pairs['pred_match_pair'].astype(int)
  
  print(f'At threshold {err_threshold} accuracy score is {accuracy_score(test_pairs.match_pair.values, test_pairs.pred_match_pair.values):.4f}')
