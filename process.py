import tensorflow as tf

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

def create_bottlenecks(sess, d, bottleneck_root, jpeg_data_tensor, bottleneck_tensor, box_ext='_box.npy', lab_ext='_lab.npy'):
  idl = os.path.join(d,'annotation.idl')
  folder_name = os.path.basename(os.path.normpath(d))
  with open(idl) as f:
    entries = f.read().split(';\n')
    for entry in entries:
      if entry:
        im_path, bbox = entry.split(': ')
        im_path = im_path.replace('"', '')
        im = os.path.join(d,im_path)

        image_data = gfile.FastGFile(im, 'rb').read()
        try:
          bottleneck_values = run_bottleneck_on_image(
              sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        except:
          raise RuntimeError('Error during processing file %s' % im_path)

        h_box = 480/8.
        w_box = 640/8.
        bbox = make_tuple(bbox)
        bottleneck_labels = np.zeros((8,8), dtype=np.float32)
        for box in bbox:
          x1,y1 = box[0], box[1]
          x2,y2 = box[2], box[3]
          bottleneck_labels[int(np.floor(y1/h_box)):int(np.ceil(y2/h_box)),int(np.floor(x1/w_box)):int(np.ceil(x2/w_box))] = 1

        fn, ext= os.path.splitext(im_path)
        bv_path = os.path.join(bottleneck_root, folder_name, fn + box_ext)
        bl_path = os.path.join(bottleneck_root, folder_name, fn + lab_ext)
        np.save(bv_bath, bottleneck_values, allow_pickle=True)
        np.save(bl_path, bottleneck_labels, allow_pickle=True)

def get_image(d, entry):
    im_path, _= entry.split(': ')[0].replace('"','')
    fn, ext= os.path.splitext(im_path)
    return os.path.join(d, fn)

def create_image_lists(annot_path):
  dirname = os.path.dirname(annot_path)
  folder_name = os.path.basename(os.path.normpath(d))
  bottleneck_dirname = os.path.join(FLAGS.bottleneck_root, folder_name)

  training_images = []
  testing_images = []
  validation_images = []

  with open(annot_path) as f:
    entries = f.read().split(';\n')
    images = map(lambda entry : get_image(bottleneck_dirname, entry), entries)
    for image in images:
      base_name = os.path.basename(file_name)
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)

  result[label_name] = {
      'training': training_images,
      'testing': testing_images,
      'validation': validation_images,
  }
  return result
