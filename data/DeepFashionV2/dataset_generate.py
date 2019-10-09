import os


if __name__ == '__main__':
    root_path = '/home/hxcai/Work/data/deepfashion2_retrieval'

    valid_category_ids = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    # valid_category_ids = [1, 2, 3, 4, 5, 6]

    data_set = 'validation'
    with open('{}.txt'.format(data_set), 'w') as set_writer, \
            open('{}_label.txt'.format(data_set), 'w') as label_writer, \
            open('{}_query.txt'.format(data_set), 'w') as query_writer, \
            open('{}_gallery.txt'.format(data_set), 'w') as gallery_writer, \
            open('{}_query_id.txt'.format(data_set), 'w') as query_id_writer, \
            open('{}_gallery_id.txt'.format(data_set), 'w') as gallery_id_writer:
        id = 0
        for category_id in os.listdir(os.path.join(root_path, data_set)):
            cid = int(category_id)
            if cid in valid_category_ids:
                if cid < 7:
                    label = cid -1
                else:
                    label = cid - 4
            else:
                continue

            for pair_id in os.listdir(os.path.join(root_path, data_set, category_id)):
                for style in os.listdir(os.path.join(root_path, data_set, category_id, pair_id)):
                    images = list(os.listdir(os.path.join(root_path, data_set, category_id, pair_id, style)))
                    images = [os.path.join(data_set, category_id, pair_id, style, name) for name in images]
                    query_images = [name for name in images if 'user' in name]
                    gallery_images = [name for name in images if 'shop' in name]
                    if len(images) > 2:
                        set_str = '\t'.join(images)
                        label_str = str(label)
                        set_writer.write('{}\n'.format(set_str))
                        label_writer.write('{}\n'.format(label_str))

                    if len(gallery_images) > 0:
                        for name in query_images:
                            query_writer.write('{}\n'.format(name))
                            query_id_writer.write('{}\n'.format(id))
                        for name in gallery_images:
                            gallery_writer.write('{}\n'.format(name))
                            gallery_id_writer.write('{}\n'.format(id))

                        id += 1
