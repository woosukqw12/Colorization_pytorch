# Colorization_pytorch
Unet, GAN

## Image Colorization Project

## Index

1. [Start](https://www.notion.so/e890c5a01165470793087e3d20f584fa?pvs=21)
2. [Index](https://www.notion.so/e890c5a01165470793087e3d20f584fa?pvs=21)
3. [Background](https://www.notion.so/e890c5a01165470793087e3d20f584fa?pvs=21)
4. [Data Setup](https://www.notion.so/e890c5a01165470793087e3d20f584fa?pvs=21)
5. [Model](https://www.notion.so/e890c5a01165470793087e3d20f584fa?pvs=21)
6. [Analysis](https://www.notion.so/e890c5a01165470793087e3d20f584fa?pvs=21)
7. [Follow-up](https://www.notion.so/e890c5a01165470793087e3d20f584fa?pvs=21)

## 1. Start

## 2. Index

## 3. Background

- Task: image colorization
    - What is it?
        - colorizing black and white images
        - used to require a huge amount of human input → now can be fully done with the power of AI and deep learning
    - channel grayscale input → channel a, b output (L*a*b color space)
- Color space: RGB vs L*a*b
    - RGB: 3 numbers indicating how much Red, Green, Blue the pixel has
        - Red Channel, Green Channel, Blue Channel
        
        ![RGB 예시 사진](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1db8893a-87a0-4edc-b142-23196a09607b/Untitled.png)
        
        RGB 예시 사진
        
    - L*a*b:
        - Lightness Channel: encodes the lightness of each pixel
        - *a channel: encodes the green-red of each pixel
        - *b channel: encodes the yellow-blue of each pixel
            
            ![Lab color space ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0dc48766-2190-49d2-823c-1663230b7668/lab_color1.png)
            
            Lab color space 
            
        
        ![Lab 예시 사진](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dba383fb-2356-455d-a9da-eec09e7e9a2e/Untitled.png)
        
        Lab 예시 사진
        
    - L*a*b color space has advantage over RGB color space when training the models for colorization problem
        1. grayscale image should be given to the model when training it for colorization
            - RGB: has to convert the image to grayscale
            - L*a*b: can feed L channel of the image without additional work
            
            —> RGB requires additional converting process
            
        2. model has to predict remaining color channels based on the grayscale image
            - RGB: has to predict 3 numbers (Red, Green, Blue channel)
            - L*a*b: only have to predict 2 numbers since L channel is already available (*a, *b channel)
            
            —> predicting RGB channel is a more diffcult and unstable task due to the higher number of combinations it requires
            

## 4. Data Setup

- used dataset:
    - MIRFLICKR 25k(used for pre-train)
        
        ![kaggle에서 위 데이터셋을 가공하여, L.npy, ab.npy형태의 데이터셋을 제공한 것을 사용함.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/177c4d77-06b5-402e-8e11-8c7064487967/Untitled.png)
        
        kaggle에서 위 데이터셋을 가공하여, L.npy, ab.npy형태의 데이터셋을 제공한 것을 사용함.
        
    - COCO(used for main-train)
        
        ![우리 코드에선 A sample of the coco dataset for object detection을 사용했고, 총 21837개 이미지의 데이터셋이라는 설명 있으면 될듯?](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3b32ca3-abdd-47b0-bd0d-b664ccdf5a59/Untitled.png)
        
        우리 코드에선 A sample of the coco dataset for object detection을 사용했고, 총 21837개 이미지의 데이터셋이라는 설명 있으면 될듯?
        
- structure of dataloader.
    - dataloader for MIRFLICKR 25k
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/64ccde76-380e-49a8-92b1-967920048ee3/Untitled.png)
    
    - dataloader for COCO
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ab49bcb2-012a-410c-b203-8ea4fedb8edc/Untitled.png)
    

## 5. Model

1. pre-trained DynamicUnet (backbone: resnet18)
    - Dynamic Unet
        - Unet: Fully-Convolutional Network based model designed specifically for image segmentation. Can be divided into Contracting Path and Expanding Path. Contracting Path captures the context of the image. Expanding Path goes through upsampling of the context map to get high-resolution segmentation result.
            
            ![U-Net 기본 구조 (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4be4716c-7368-4918-8f8f-b754020b468d/Untitled.png)
            
            U-Net 기본 구조 (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            
        - Dynamic Unet: a specific type of U-Net that is used to increase the accuracy of the model
            - Contracting Path: during the downsampling path, ResNet is used
            - Expanding Path: During the upsampling path, technique called Pixel Shuffle is used. This is used to reduce the checkerboard artifacts(체커무늬 흔적) that may form in the image due to the concatenation of two layers.
            - 아래 논문 261 페이지 마지막 문단 참조
            
            [Design of Intelligent Applications using Machine Learning and Deep Learning Techniques](https://books.google.co.kr/books?id=VHM3EAAAQBAJ&pg=PA261&lpg=PA261&dq=dynamic+unet&source=bl&ots=lUoIvTPhxz&sig=ACfU3U1q_RaCSSM-4Pc84Bg4CeuRbahkeA&hl=en&sa=X&ved=2ahUKEwiJ2cHYyOX7AhU7xYsBHXOJAtMQ6AF6BAgiEAM#v=onepage&q=dynamic%20unet&f=false)
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f3da500a-0026-4d11-a734-8764446686f6/Untitled.png)
            
    - brief about pre-train & why pre-train?
        
        : method of initializing weights of a model that were initialized to an arbitrary value into weights learned in another task
        
        - it effectively initializes weights and biases → can effectively train several hidden layers
        - since it doesn’t requires labeled traing data, unsupervised learning is possible → can train the model with large dataset
    - how to do?
    
    ![adam algorithm - SGD+Momentum + AdaGrad = Adam이라 마땅한 그림이 없어서 이거 넣어뒀어 마침 L1 loss도 수식이니까 그냥 깔끔st로 넣고 이걸로 학습했어요 해도 될듯??](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/49ca7709-53ef-47e3-bc81-d962de9fb439/Untitled.png)
    
    adam algorithm - SGD+Momentum + AdaGrad = Adam이라 마땅한 그림이 없어서 이거 넣어뒀어 마침 L1 loss도 수식이니까 그냥 깔끔st로 넣고 이걸로 학습했어요 해도 될듯??
    
    ![일단 gif찾아둠.. 근데 노션 개인요금제 5MB제한땜에 안예쁜거 올려놨어. 그냥 구글에 adam optim gif 검색하면 예쁜거 많아서 그거 써도 좋을것같에](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0c2e703b-c910-4fdc-80c9-2730417ce623/tem.gif)
    
    일단 gif찾아둠.. 근데 노션 개인요금제 5MB제한땜에 안예쁜거 올려놨어. 그냥 구글에 adam optim gif 검색하면 예쁜거 많아서 그거 써도 좋을것같에
    
    ![L1loss](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dbe498f6-cae6-417c-b47e-7324bba984e1/Untitled.png)
    
    L1loss
    
    - pre-train result ( briefly )
    학습 완료되면 loss나온 사진 들어갈 예정
    
    ![이게 전체 결과긴 한데, 너무 길다..](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5259397c-1ace-4831-8197-483fb6f6e9e1/Untitled.png)
    
    이게 전체 결과긴 한데, 너무 길다..
    
    ![이건 그림판 편집본 ← 요거 쓰는게 그나마 더 잘보일..듯?..](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5906bc16-4805-4412-97cd-20b11cce93df/Untitled.png)
    
    이건 그림판 편집본 ← 요거 쓰는게 그나마 더 잘보일..듯?..
    
- main-training

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/144cdb26-5b62-48fd-881c-1d1e98b60645/Untitled.png)

- training process

1. make fake_color
2. Unet train
3. Unet backpropagation
4. Unet optimizer optimization

- test results
    
    ![5 epoch result imgs](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6620bd69-4235-4626-9ded-31f48db425d4/output0.png)
    
    5 epoch result imgs
    
    ![20 epoch result imgs](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e0b4b05e-3d00-48d8-b845-2c19d8f9c2b7/output3.png)
    
    20 epoch result imgs
    
    ![40 epoch result imgs](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d75ad277-1e87-40c8-b563-a8e9f4ede37c/output7.png)
    
    40 epoch result imgs
    
    ![60 epoch result imgs](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36645442-53c0-45fa-9ecc-72911ea0a6b6/output11.png)
    
    60 epoch result imgs
    
    ![5 epoch loss plot](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32dec6b3-1535-4390-bbbe-5d0f6ccd790e/plot0.png)
    
    5 epoch loss plot
    
    ![20 epoch loss plot](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2800bc88-6f01-4c59-a1af-c1112e2dea0f/plot3.png)
    
    20 epoch loss plot
    
    ![40 epoch loss plot](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38637138-a25e-448d-b554-b8800798281f/plot7.png)
    
    40 epoch loss plot
    
    ![60 epoch loss plot](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0dc80a60-c029-4452-a9e3-eeef082dfa77/plot11.png)
    
    60 epoch loss plot
    
1. GAN(G: Unet, D: CNN)
    - about GAN
    
    ![ref: https://m.blog.naver.com/euleekwon/221557899873](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/55b5751f-5efd-4084-ab98-10d9bfbeeabe/Untitled.png)
    
    ref: https://m.blog.naver.com/euleekwon/221557899873
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06ad858e-f6cb-4f96-9e69-290de8fb22a9/Untitled.png)
    
    GANs are algorithm architectures that use two NNs, competing with each other to become more accurate in their predictions.
    
    - structure
    
    ![generator](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8c9f8f51-c649-44f2-9b33-4bb28df1311d/Untitled.png)
    
    generator
    
    ![discriminator](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1074454a-c84b-4a91-8094-2421bc91ba15/Untitled.png)
    
    discriminator
    
     
    
    - how to learn?
    
    ![gan loss —> D의 loss는 최대화, G의 loss는 최소화하는 방향으로 학습함.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb52f4c3-96f3-45ca-8ead-eabcb97bdd32/Untitled.png)
    
    gan loss —> D의 loss는 최대화, G의 loss는 최소화하는 방향으로 학습함.
    
    ![GAN](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5d7bc454-11ae-4374-81e8-ca0fa8f7c81f/Untitled.png)
    
    GAN
    
    - learning process
    1. make fake_color
    2. Discriminator
        1. Discriminator train
        2. Discriminator backpropagation
        3. Discriminator optimizer optimization
    3. Generator
        1. Generator train
        2. Generator backpropagation
        3. Generator optimizer optimization
    
    ** 2~4, 5~7을 다른 방식으로 묶어서 표현해도  괜찮을듯 2-1, 2-2, 2-3 느낌으로 (반영함)
    
    - test results
        
        ![1 epoch result imgs](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1f79bf9-b53a-43e2-9910-53b2a73c24f2/output3.png)
        
        1 epoch result imgs
        
        ![10 epoch result imgs](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a6bdd84-aee4-4130-a375-e78daeb52e9a/output_10epoch.png)
        
        10 epoch result imgs
        
        ![20 epoch result imgs](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5827966c-560c-45dc-9e6d-3e606d47a339/output_20epoch.png)
        
        20 epoch result imgs
        
        ![GAN loss plot - 20 epoch](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e637e5b3-a1ce-4056-bada-56294828e9bc/gan_20epoch_plot.png)
        
        GAN loss plot - 20 epoch
        
- our photo results

![우리가 직접 찍은 사진으로 테스트한 결과 ( 위-gray_scale, 중간-fake_img, 아래-real_img)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0cb2fbc2-2db7-4fcb-87a7-46764ea2d541/Untitled.png)

우리가 직접 찍은 사진으로 테스트한 결과 ( 위-gray_scale, 중간-fake_img, 아래-real_img)

## 6. Analysis

- compare results between  DynamicUnet and GAN

![60 epoch result imgs - Unet](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36645442-53c0-45fa-9ecc-72911ea0a6b6/output11.png)

60 epoch result imgs - Unet

![20 epoch result imgs - GAN](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5827966c-560c-45dc-9e6d-3e606d47a339/output_20epoch.png)

20 epoch result imgs - GAN

## 7. Follow-up

- StyleGAN-ADA

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb39f589-c91a-4df1-9c37-693e6fd78654/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/588698f3-794e-491f-af2f-4511afe81fb2/Untitled.png)

Example generated images for several datasets with limited amount of training data, trained using ADA.
