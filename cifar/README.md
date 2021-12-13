# TTT++ CIFAR-10/100

TTT++ on the CIFAR-10/100 under [common corruptions](https://github.com/hendrycks/robustness) or [natural shifts](https://arxiv.org/abs/1806.00451).

### Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

To download datasets:

```bash
export DATADIR=/data/cifar
sudo mkdir -p ${DATADIR} && cd ${DATADIR}
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
wget -O CIFAR-100-C.tar https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
```

### Pre-trained Models

The checkpoint of the pre-train Resnet-50 can be downloaded (214MB) using the following command: 

```bash
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
gdown https://drive.google.com/uc?id=1TWiFJY_q5uKvNr9x3Z4CiK2w9Giqk9Dx && cd ../..
mkdir -p results/cifar100_joint_resnet50 && cd results/cifar100_joint_resnet50
gdown https://drive.google.com/uc?id=1-8KNUXXVzJIPvao-GxMp2DiArYU9NBRs && cd ../..
```

### Test-time Adaptation

Our proposed TTT++:

```bash
bash scripts/run_ttt++_cifar10.sh
```

Prior method TENT:

```bash
bash scripts/run_tent_cifar10.sh
```

Prior method SHOT:

```bash
bash scripts/run_shot_cifar10.sh
```

The scripts above yield the following results (classification errors) on the Cifar10-C under the snow corruption:

| Method | Error (%) |
|:------:|:---------:|
|  Test  |   21.93   |
|  [TENT](https://openreview.net/forum?id=uXl3bZLkr3c)  |   12.24   |
|  [SHOT](https://proceedings.mlr.press/v119/liang20a.html)  |   13.89   |
| [TTT++](https://papers.nips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html)  | **9.79**  |

### Batch-queue Decoupling

To run our test-time algorithms with a dynamic queue for moment estimate, set the `queue size` larger than the `batch size`:

```bash
bash scripts/run_ttt++_cifar100.sh <domain> <method> <batch-size> <queue-size>
```

For instance, the line below is to run the test-time feature alignment on the Cifar100-C under the snow corruption, with a batch size of 128 and a queue size of 1024. 
```bash
bash scripts/run_ttt++_cifar100.sh snow align 128 1024
```

The script above yields the following results with different sample sizes (i.e., batch size x # batch):

| Sample Size | Error (%) |
|:-----------:|:---------:|
|   128 x 1   |   38.30   |
|   256 x 1   |   36.90   |
|   512 x 1   |   36.05   |
|   128 x 4   |   36.66   |
|   128 x 8   |   35.32   |
|   128 x 16  |   34.62   |

### Feature Visualization

To generate the t-SNE figures for feature visualization, add `--tsne` to the above bash scripts.

### Acknowledgements

Our code is built upon the public code of the [TTT](https://github.com/yueatsprograms/ttt_cifar_release) as well as that of [SHOT](https://proceedings.mlr.press/v119/liang20a.html) and [TENT](https://openreview.net/forum?id=uXl3bZLkr3c).