# Image Super Resolution

Super Resolution(SR) belongs to low-level task in CV.

For image SR, we aim to super resolve the low resolution input to high SR output.

![image](https://user-images.githubusercontent.com/36061421/119450479-87f18b80-bd66-11eb-8b56-5df25f6f2556.png)

Image above shows the process of SR, which also presents ill-posed feature of this technique. This feature comes from information deficiency. Obviously, we need to restore more information than information provided in the original image. Actually, SR is a kind of techniques of Image Restoration, whose goal is deblurring, denoising and so on. Image Restoration is different from Image Enhancement. To a certain extent, they show the same effect, which is improving the quality of an image. But Image Enhancement focuses more on Contrast Enhancement rather than deblurring like Image Restoration.

As for the history of SR, to the best of my knowledge, I cannot assert who is the first one coming up with this idea. But I can say [Thomas Shi-Tao Huang](https://grainger.illinois.edu/about/directory/faculty/t-huang1) is an influential pioneer in this field.

Image SR can still be divided into many fields. Two of them are General SR and Face SR. General SR means the task comes from different fields, we do not need to be specific. However, Face SR requires us to focus on face only. It is necessary for us to study it as a new branch because facial information is of great significance for identification and face contains more prior information which can be utilized to accomplish facial image restoration.

At last, I list some important benchmark datasets.

General SR: Set 5, Set 14, Urban 100, BSD 100...

