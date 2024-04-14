# Deep Learning - Reproducibility

## Introduction

In this blog, a reproduction is done of the paper ‘**NeRV: Neural Representations for Videos’** by Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser-Nam Lim and Abhinav Shrivastava is reproduced for our Deep learning course. The repdoruction is done on 4 different topics: Reproduction of the results presented in the paper, ablation study, hyperparameter tuning and finaly the model is tested on a new dataset.  

The paper showed that it is possible to create a Neural network that was able to reproduce video’s accurately, while being represented by functions. What is a video? Typically, a video captures a dynamic visual scene using a sequence of frames. Each frame has a raster of images, with each frame a corresponding value per pixel. The original researches show that we can think about each pixel value as a 2D curve: for each time instance *x,* there is a correlated value *y,* representing the pixels color value. This leads to their main claim: *can we represent a video as a function of time?*

---

The neural representation of videos can be beneficial for different reasons. In medical imaging, privacy can be on of the reasons a video cannot be encoded in a conventional frame to frame format. While this seams to be a great benefit, the more technical gains lay in the representation of the video. Where a normally encoded video is represented by frames, in a grid of pixels, each with different values, the videos that are encoded5 by a neural net are represented by functions: see figure 1 for a visual representation.

![Figure (1): Difference between explicit representation and neural implicit representations for videos. ](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled.png)

Figure (1): Difference between explicit representation and neural implicit representations for videos. 

Having functions that represent the overall characteristics and the behaviour, gives us the ability to interpolate between frames, creating the abilities to make high framerate slomo videos. Another pre is the ability to upscale images, since the interpolation between the function can be done more accurate. 

The researchers of the paper showed while researching NeRV's versatility, they found some intriguing applications. A key area of interest is its use in video compression. Traditional video compression methods can get quite complicated – they involve specifying key frames, estimating residual information, block-sizing video frames, and applying discrete cosine transform on image blocks. This extensive process also results in a complex decoding task. 

However, with NeRV, things become much simpler. As NeRV uses a neural network for video encoding, it is possible to treat video compression as a model compression problem. This allows to easily utilize any recognized or advanced model compression algorithm to achieve excellent compression ratios. It's a simpler, more efficient approach to video compression.

# Reproduction

The first task of this project was to reproduce the results of the third line of Table 2. For the training speed, encoding time and the decoding FPS, absolute values were taken to reproduce the results, as it was outside the scope of the project to run the SIREN and the NeRF model. 

![Figure (2)](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%201.png)

Figure (2)

The following results were achieved when trying to reproduce the numbers from above. This was done for the NeRV-S model. By definition the amount of parameters is 3.2M, just like in the table. The training speed (time/epoch) is 13.79 seconds. With 1200 epochs this adds up to a Encoding time (total training time) of 4:35:54 as can be seen from the image below. 

![Figure (3)](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%202.png)

Figure (3)

A Peak Signal to Noise Ratio (PSNR) of 34.03 was achieved during the training, which is similar enough to the value of 34.21 from the table to assume the training process was done correctly. The small difference between the values can be accounted to a slightly different model being trained which is not very unlikely in a model using 3.2M parameters. For the Decoding FPS the evaluation was run. The following results were found:

![Untitled](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%203.png)

Where it can be seen that the FPS is much lower than the FPS from Table 2. This has most probably to do with the GPU on which it was run. The GPU it was ran on is the “Nvidia Tesla P100”, available in Kaggle and the GPU used in the paper is The Nvidia RTX 2080 TI. Even though the P100 GPU has more bandwidth and memory, it is still being outperformed by the GPU in terms of clock speed, processing unit and theoretical performance. For high performance GPU tasks this can make a large difference.

# Ablation

To perform a ablation study we have looked at the influence of the NeRV layers and tested what would happen if we removed or added some of of them. We also tested the performance of the network when adding and removing a fully connected layer to see if a deeper embedding would influence the results.  Below the resulting architectures and metrics will be presented and discussed.

### Architectures tested

1. Original Architecture

```jsx
(stem): Sequential(
    (0): Linear(in_features=80, out_features=512, bias=True)
    (1): SiLU(inplace=True)
    (2): Linear(in_features=512, out_features=3744, bias=True)
    (3): SiLU(inplace=True)
  )
  (layers): ModuleList(
    (0): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 650, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=5)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (1): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (2-4): 3 x NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
  )
  (head_layers): ModuleList(
    (0-3): 4 x None
    (4): Conv2d(96, 3, kernel_size=(1, 1), stride=(1, 1))
```

1. Removal of 2 NeRV blocks
    
    We removed two of the total 5 NeRV blocks that are used in the original architecture. To still output the same resolution of 720p the up scaling factor in the second and final NeRV block has been set to 4 instead of 2. 
    

```jsx
(stem): Sequential(
    (0): Linear(in_features=80, out_features=512, bias=True)
    (1): SiLU(inplace=True)
    (2): Linear(in_features=512, out_features=3744, bias=True)
    (3): SiLU(inplace=True)
  )
  (layers): ModuleList(
    (0): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 650, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=5)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (1): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=4)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (2): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=4)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
  )
  (head_layers): ModuleList(
    (0-1): 2 x None
    (2): Conv2d(96, 3, kernel_size=(1, 1), stride=(1, 1))
  )
```

1. Addition of several NeRV Blocks, again the final resolution is 720p. 

```jsx
(stem): Sequential(
    (0): Linear(in_features=80, out_features=512, bias=True)
    (1): SiLU(inplace=True)
    (2): Linear(in_features=512, out_features=3744, bias=True)
    (3): SiLU(inplace=True)
  )
  (layers): ModuleList(
    (0): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 650, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=5)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (1): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=1)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (2): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (3): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=1)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (4): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (5): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=1)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (6): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (7): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=1)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (8): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (9): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=1)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
  )
  (head_layers): ModuleList(
    (0-3): 4 x None
    (4): Conv2d(96, 3, kernel_size=(1, 1), stride=(1, 1))
  )

```

1. Removal of Fully connected layer

```bash
Generator(
  (stem): Sequential(
    (0): Linear(in_features=80, out_features=3744, bias=True)
    (1): SiLU(inplace=True)
  )
  (layers): ModuleList(
    (0): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 650, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=5)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (1): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (2-4): 3 x NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
  )
  (head_layers): ModuleList(
    (0-3): 4 x None
    (4): Conv2d(96, 3, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

1. Addition of a fully connected layer

```bash
Generator(
  (stem): Sequential(
    (0): Linear(in_features=80, out_features=512, bias=True)
    (1): SiLU(inplace=True)
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): SiLU(inplace=True)
    (4): Linear(in_features=512, out_features=3744, bias=True)
    (5): SiLU(inplace=True)
  )
  (layers): ModuleList(
    (0): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 650, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=5)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (1): NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(26, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
    (2-4): 3 x NeRVBlock(
      (conv): CustomConv(
        (conv): Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_scale): PixelShuffle(upscale_factor=2)
      )
      (norm): Identity()
      (act): SiLU(inplace=True)
    )
  )
  (head_layers): ModuleList(
    (0-3): 4 x None
    (4): Conv2d(96, 3, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

### Results

In the table below the results of the different architectures can be seen. For the ablation study of the fully connected layers also 1200 epoch training runs have been performed to get a better understanding of training times. The results of removal and addition of NeRV layers showed expected behavior in the results. The training time would increase when adding extra NeRV Blocks and decrease when removing NeRV layers. One important observation that we made is that the number of parameters increased when removing NeRV blocks while the training time decreased. This was due to the fact that we increased the upscale_factor of the pixelshuffle layers in the NeRV blocks to get the same resolution output of 720p. This increased the number of parameters but the training time was lower. This probably indicates that the pixel shuffle is a less computational expensive operation than the standard 2d convolution operation. The second important observation we made is that the fully connected layers do not add extreme value to the model while increasing training time. In table column 4 and 5 we can see that for the 1200 epoch runs, the training time, when removing the fully connected layer decreased by 1 hour. However the metrics only decreased a little bit. It might therefore be wise to only use an embedding layer without other fully connected layers as this does decrease performance only a little bit while decreasing training time significantly.  

| Title | Column 1 | Column 2 | Column 3 | Column 4 | Column 5 |
| --- | --- | --- | --- | --- | --- |
|  | Original (git)
300/1200 epoch | -2 NeRV 
300 epoch | +4 NeRV
300 epoch | -1 FC 
300 / 1200 epoch | +1 FC
300 / 1200 epoch |
| PSNR | 32.13 / 33.98 | 30.28 | 33.62 | 31.90 / 33.55 | 31.95 / 33.93 |
| MSSSIM | 0.9591 / 0.9734 | 0.9370 | 0.9745 | 0.9564 / 0.9704 | 0.9571 / 0.9731 |
| Training time | 01:16:43 / 05:10:35 | 01:02:17 | 01:31:32 | 01:15:15 / 03:59:36 | 01:15:50 / 05:03:55 |
| Parameters | 3.2019005M | 3.804785M | 5.200975M | 1.543025M  | 3.464561M |

# Hyper Parameter Tuning

The process of hyperparameter involves the tuning of the parameters that are not directly involved in the model. Hyperparameters are the parameters that are not learned during the training process, but are set beforehand. There are a few different techniques to do this, such as a grid search, random search or bayesian optimisation. In this blog post, we will look for the most suitable technique to find an optimal set of hyperparameters. 

### RayTune

Tune is a python library for experiment execution and hyperparameter tuning at any scale.  I will be refering to it as Ray Tune, since it is part of the larger, open-source framework Ray. This framework is made for scaling AI and Python applications.

Ray Tune offers more then just a simple grid search, and allows for distributed computation of the hyperparameters, ranging from multiple GPU’s to multiple machines. In addition to using a simple grid search algorithm, Ray Tune also offers more sophisticad algorithms like random search, Bayesian optimisation and population based training. 

Another reason to use a library like Raytune is the early stopping features; their ASHA scheduler terminates uneffective trials, to minimize the training time.

So, enough reasons to implement an algorithm like this! To do this, we followed the ‘[Hyperparameter tuning with Ray Tune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)’ guide on the website on pytorch. This shows us that we need to implement a slightly modified training function, and a main function where we define the search space, what kind of scheduler to use and give the model parameters. 

We want to start with a simple test, with the only tunable parameter being the batch size. So to `def main()` we add:

```python
    config = {
        "batch_size": tune.choice([1, 2, 4, 8]),
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 1, "gpu": args.ngpus_per_node},
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
    )
```

Since the dataloader is already built into our model, we only need to modify our model. In to original code, the model is generated by the function `Generator()` ; we can leverage this function and give it the parameters of the Ray Tune search space:

```python
args.batchSize = config["batch_size"]
```

With a simple modification like this, we push the code to GIT, and clone the code in Kaggle. However, this is where the problems begin: the installation of Ray tune does not work completely properly, resulting in some pip errors. However, we just try and give it a go!

![Figure (4)](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%204.png)

Figure (4)

When trying to run train_nerv.py, the problems arise: the raytune package does not seem to work properly on the Kaggle server, resulting in the error codes:

![Figure (5)](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%205.png)

Figure (5)

While there are some examples of Ray Tune working in Kaggle, we could not get it to work properly, and the solution reaches beyond the scope of this project. Examples where they fixed this problem show more low level linux commands, for which we do not have the acces on the Kaggle server. 

### Grid Search

This is why we decided to run a few different runs on kaggle with different parameters for the grid search. We have chosen to tune 4 parameters:

- Batch size
    - Trial and error gives that the largest possible batch size is around 10 samples per batch; otherwise we run out of memory on the GPU.
- Learning Rate
    - The learning rate is related to the step size in the optimisation algorithm. The learning rate is standard 0.0002, and is changed in the range 0.0001 ~ 0.002
- Loss type
    - There are different loss types: for this grid search we took into account the L2 (Mean squared Error) loss, Cross entropy loss and Mean absolute loss.
- Warmup epochs
    - During the warm-up phase, the learning rate may start at a very small value and gradually increase to its intended value.

For each parameter, we have picked a few different values. First, each parameter is run seperately; afterwards we pick the best combinations, and see if they have an influence on each other. The MSSNIP is the key value on which the choices are based. 

| Parameter | Value 1 | Value 2 | Value 3 | Value 4 | Value 5 |
| --- | --- | --- | --- | --- | --- |
| Batch Size:
Learning rate:
MMSIM:
Runtime: | 1
0.0002
0.9590
1:11:04 | 2
0.0002
0.9385
1:09:00 | 4
0.0002
0.9077
1:10:03 | 8
0.0002
0.8686
1:09:57 | 10
0.0002
0.8306
1:11:01 |
| Batch Size:
Learning Rate:
MMSIM:
Runtime: | 1
0.0001
0.8813
1:10:18 | 2
0.0005
0.9385
1:08:10 | 4
0.001
0.9428
1:09:00 | 8
0.002
0.9396
1:08:47 |  |
| Loss Type:
Batchsize:
Learning rate:
MMSIM:
Runtime:
 | L2 - MSE
1
0.0001 
0.9590
1:11:04 | SSIM
1
0.001
0.9228
1:10:04 | L1 - MA
1
0.0001 
0.8779
1:04:27 |  |  |
| Warmup Epochs:
Batchsize:
Learning Rate MMSIM:
Runtime: | 0
1
0.00002
0.8875
1:05:09 | 0.1
1
0.0002
0.8888
1:05:19 | 0.2
1
0.0002
0.9590
1:11:04 | 0.5
1
0.0002
0.8872
1:05:07 |  |

## Conclusion

As can be seen, the hyperparameters have been tuned quite well. However, an interesting observation is that when the batchsize increases, the performance is going down. However, when the learning rate is also increased with the batchsize, the performance goes back to the old levels. This however, does not decrease runtime; there is no real reason to do this. 

# New Dataset

To evaluate the model's performance on a dataset distinct from the "Big Bug Bunny" dataset, it is applied to a video from the Cholec80 dataset. This dataset comprises 80 videos of cholecystectomy surgeries, conducted by 13 surgeons, and recorded at a frame rate of 25 frames per second (fps). For the purpose of this test, only the first 10 seconds of one video from the Cholec80 dataset are utilized, ensuring that the number of frames available for model training is comparable to that of the training set used with the "Big Bug Bunny" dataset. The latter dataset involved training on 132 frames at a resolution of 720x1080. In contrast, the selected video from the Cholec80 dataset encompasses 132 frames, with a resolution of 854x480. Due to storage constraints on our GitHub page, only the smaller model variant will be subjected to testing with the Cholec80 dataset, as the other two model sizes exceed our storage capacity. 

### Experimental setup overview:

In our experimental setup, we utilized the same hyperparameters as in the paper. All experiments were conducted on Kaggle using the GPU P100 accelerator, maintaining the hyperparameters as specified in the original paper. For the Kaggle experiment focused on the "Big Buck Bunny" dataset, we trained the model for 300 epochs, employing an upscale factor identical to that used for the "Big Buck Bunny" set. The Model is trained for 300 epochs, because of computational resources. Below is an organized overview of the experimental parameters.

| Parameter/Setting | Value/Description |
| --- | --- |
| Optimization Algorithm | Adam optimizer  |
| Initial Learning Rate | 5e-4 |
| Learning Rate Schedule | Cosine annealing |
| Batch Size | 1 |
| Training Epochs | 300 for Kaggle experiment |
| Warmup Epochs | 30 |
| Model Architecture | NeRV with varying NeRV blocks and up-scale factors |
| NeRV Blocks and Up-Scale Factors | 5 blocks with scales 5, 2, 2, 2, 2 (720p) |
| Input Embedding Parameters (b, l) | b = 1:25, l = 80 |
| Evaluation Metrics | PSNR, MS-SSIM  |
| Cholec 80 datasets |  Cholec80 (first 10s of one video, 132 frames, 854x480 resolution) |
| GPU Accelerator | P100 (Kaggle) |
| Upscale Factor (Kaggle Experiment) | 5, 2, 2, 2 |
| GitHub Storage Constraints | Only smaller model variant tested with Cholec80 due to size limitations |
| Loss objective alpha | 0.7 |

### Results

Due to the upscale factors the resolution of the images went from 854x480 to 1280x720. The training time using the kaggle GPU 100 accelerator for the 132 frames is around 1:10:34, this is roughly the same as for the training time of the model on the big bug bunny dataset which was around 1:15:00.

### Training data

![Figure (6);  PSNR vs epochs](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%206.png)

Figure (6);  PSNR vs epochs

![Figure (7); MSSSIM vs epochs](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%207.png)

Figure (7); MSSSIM vs epochs

### Visualizations

![Figure (8); Ground truth of frame 1.](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%208.png)

Figure (8); Ground truth of frame 1.

![Figure (9); Prediction of frame 1.](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%209.png)

Figure (9); Prediction of frame 1.

![Figure(10); Ground truth of frame 30.](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%2010.png)

Figure(10); Ground truth of frame 30.

![Figure(11); Prediction of frame 30.](Deep%20Learning%20-%20Reproducibility%2014b12844bbc84d74b817fdd05dd092b7/Untitled%2011.png)

Figure(11); Prediction of frame 30.

### Evaluation

Figures 3 and 4 illustrate the model's performance across epochs, as evidenced by increases in PSNR (Peak Signal-to-Noise Ratio) and MSSSIM (Multi-Scale Structural Similarity Index), aligning with our expectations from the paper. The model's average PSNR is recorded at 29.26, while the MSSSIM stands at 0.9146. The paper stated that the model  trained on the Big Buck Bunny dataset over 300 epochs, the model achieved a PSNR of 32.21. This represents a difference of 2.95 in PSNR compared to the our model, translating to a relative difference of approximately 9.15% against the Big Buck Bunny benchmark. This difference could be due to alot of factors like **Dataset Complexity and Content**: The inherent complexity and content differences between the datasets could significantly affect the model's performance. The Big Buck Bunny dataset contains less complex scenes and has a more uniform like backgrounds, it may naturally yield higher PSNR and MSSSIM values compared to the cholec80 dataset with more complex visuals. **Hardware**: Differences in the training hardware (e.g., types of GPUs, memory bandwidth) ) can introduce variations in training efficiency and model optimization, indirectly affecting performance metrics.

### Conlusion

Visual examination of the model's output reveals a slight loss in detail within the predicted frames, although the overall frame recreation is satisfactory. The suitability of this level of detail recreation largely depends on the intended application of the videos. For instance, applications within healthcare analysis may necessitate a degree of detail comparable to the original training data, highlighting the context-dependent adequacy of the model's performance. Looking at the overall performance, visual and numerical the model showcases its capability to also work on datasets other then presented in the paper.

# References

- A.P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy, 
**1. EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos**, 
*2. IEEE Transactions on Medical Imaging (TMI)*, 
[3. arXiv preprint](http://arxiv.org/abs/1602.03012), 2017
- Chen, H., He, B., Wang, H., Ren, Y., Lim, S., & Shrivastava, A. (2021, October 26). *NERV: Neural representations for videos*. arXiv.org. https://arxiv.org/abs/2110.13903