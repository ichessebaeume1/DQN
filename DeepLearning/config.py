URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE_NAME = "fashion_mnist_images.zip"
FOLDER_NAME = "fashion_mnist_images"
delete_folder_after_unzip = True

load_checkpoint = False
checkpoint_file_name = None

generations = 10
print_every = 100

layers = 4
neurons = 128
batch_size = 128

scaling_method = 1   # 2 = 0 to 1 and 1 = -1 to 1

use_dropout = False

weight_reg = 0.0005
bias_reg = 0.0005

output_size = 10

learning_rate = 0.05
decay = 0.001

plot_gen_acc = True
plot_gen_loss = True
plot_gen_data_loss = False
plot_gen_reg_loss = False
plot_gen_lr = False

acc_in_percent = True
