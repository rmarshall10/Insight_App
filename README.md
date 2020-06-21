# JuggleWizard
[in progress]

Soccer is the most popular sport in the world, and there are millions of young, developing players around the world. However, it is difficult to get kids to practice on their own, especially in places like the US, where there is much less of a “pick-up” soccer culture than other places.
Soccer juggling is a great way to improve hand-foot coordination and foot skills which you can do on your own, or with others. Soccer juggling is where you keep the ball in the air by hitting it with your feet, or other body parts except for the hands.
This Flask app counts the consecutive juggles of a player, and determines which body part they use for each juggle. It could assign different point values to different body parts, or have “levels” where players have to juggle in certain sequences.
In a future, scaled-up version of the app, users could share with friends and teammates on social networks. 

### How It Works

A user uploads a video of juggling a soccer ball, and the output is the tracked video with the counters and a breakdown of counts with (rate metrics).
The ball is detected by a single-shot detection neural network (SSD_mobilenet_v2), pretrained on COCO image dataset, and retrained on thousands of images of soccer balls in action from ImageNet.
The pose detection model is from Tensorflow, and uses OpenCV for tracking.

### Contents


