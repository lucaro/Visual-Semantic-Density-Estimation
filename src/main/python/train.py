import torch
from torch import nn

from timm import create_model
from tqdm import tqdm

import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass


def save_ckp(epoch, model, optimizer, is_best):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    f_path = 'data/model/checkpoint_' + str(epoch) + '.pt'
    torch.save(checkpoint, f_path)
    if is_best:
        torch.save(checkpoint, 'data/model/checkpoint_best.pt')



base_model_name = "convnext_tiny_in22k"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = create_model(base_model_name, pretrained=True).to(device)

children = list(base_model.children())
relevant = children[:-1].copy()
relevant.append(children[-1][:-2])
relevant.append(nn.Linear(768, 2048, bias = True))
relevant.append(nn.GELU())
relevant.append(nn.Linear(2048, 1, bias = True))

model = torch.nn.Sequential(*relevant).to(device)


#freeze initial layers
for param in list(model.parameters())[:-4]:
    param.requires_grad = False


feature_description = {'img': tf.io.FixedLenFeature([], tf.string), 'count': tf.io.FixedLenFeature([], tf.int64)}

def _parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    example['img'] = tf.io.parse_tensor(example['img'], tf.float32)
    return example



if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).to(device)

epochs = 500
batch_size = 256

loss_fn = nn.L1Loss()
test_loss_fn = nn.L1Loss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

imgs = []
counts = []
last_best_loss = 99999

for epoch in range(epochs):

    print('training epoch', epoch)
    model.train()

    train_loss = 0
    train_count = 0

    raw_dataset = tf.data.TFRecordDataset('data/tfrecord/train.tfrecord')
    parsed_dataset = raw_dataset.map(_parse_example).shuffle(1024, reshuffle_each_iteration=True)

    for record in tqdm(parsed_dataset, total = 712299):
        try:
            img = torch.from_numpy(record['img'].numpy())

            imgs.append(img)
            counts.append([record['count'].numpy()])

            if len(imgs) >= batch_size:

                img_tensor = torch.stack(imgs).to(device)
                pred = model(img_tensor)
                y = torch.tensor(counts).to(device)

                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                imgs = []
                counts = []

                train_loss += loss.item()
                train_count += 1
        except IOError:
            print('error loading image')

    print('finished training epoch', epoch, 'training loss', train_loss / train_count)

    #test

    print('testing epoch', epoch)

    model.eval()

    test_loss = 0
    item_count = 0

    raw_dataset = tf.data.TFRecordDataset('data/tfrecord/test.tfrecord')
    parsed_dataset = raw_dataset.map(_parse_example)


    for record in tqdm(parsed_dataset, total = 137669):
        try:
            img = torch.from_numpy(record['img'].numpy())

            imgs.append(img)
            counts.append([record['count'].numpy()])

            if len(imgs) >= batch_size:
                with torch.no_grad():
                    img_tensor = torch.stack(imgs).to(device)
                    pred = model(img_tensor)
                    y = torch.tensor(counts).to(device)

                    loss = test_loss_fn(pred, y).item()

                    test_loss += loss
                    item_count += len(imgs)


                imgs = []
                counts = []
        except IOError:
            print('error loading image')


    avg_loss = test_loss / item_count

    print('tested after epoch', epoch, 'with loss', avg_loss)

    is_best = avg_loss < last_best_loss

    if is_best:
        last_best_loss = avg_loss

    if is_best or epoch % 20 == 0:
        print('store model')
        save_ckp(epoch, model, optimizer, is_best)

