from collections import OrderedDict

from tqdm import tqdm


def load_flickr_set(images, text, file, test):
    """
    Produces a ordered dict of tuples of data and label from the ids defined in file.
    Train Structure:
    {
        id: (id, img_data, word vector)
    }

    TODO: THIS IS NOT USED RIGHT NOW:
    Test Structure:
    {
        id: (id, img_data, (c1, c2, c3, c4, c5))
    }

    :param images: preproccesed data as ordered dict
    :param text: preproccesed text as pandas dataframe with columns 'image_idx', 'caption', 'caption_idx'
    :param file: Flickr_8k.*Images.txt
    :param test: indicates if we want a test dataset
    :return: ordered dict
    """
    dataset = OrderedDict()
    set_ids = read_ids(file)

    for pic_id in tqdm(set_ids, desc="Generating Flickr dataset"):
        img_data = images[pic_id]
        text_data = text[text['image_idx'] == pic_id]

        # For testing data, we want all available captions as true labels
        # result -> (id, x, (y1, y2, .., y5))
        if test:
            labels = ()
            for idx, row in text_data.iterrows():
                labels += (row['caption'],)

            dataset[pic_id] = (pic_id, img_data, labels)

        # For training data, we want to use the captions to generate new training objects
        # result -> (id, x, y)
        else:
            for idx, row in text_data.iterrows():
                new_name = pic_id + '-' + row['caption_idx']
                dataset[new_name] = (new_name, img_data, row['word2vec'])

    return dataset


def read_ids(file):
    set_ids = []
    with open(file, 'r', encoding='UTF-8') as f:
        # Read the data
        for line in f:
            # Strip the ending newline char
            set_ids.append(line.strip("\n"))

    return set_ids


def get_caption_set(text, file):
    """ Returns all captions of the pictures ids defined in file

    :param text: preproccessed captions as dataframe
    :param file: Flickr_8k.*Images.txt
    :return: list of captions
    """
    set_ids = read_ids(file)
    result = []

    for pic_id in tqdm(set_ids, desc="Generating captions training set"):
        text_data = text[text['image_idx'] == pic_id]

        for _, row in text_data.iterrows():
            result.append(row['caption'])

    return result
