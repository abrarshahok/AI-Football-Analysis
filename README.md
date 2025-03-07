# AI Football Analysis

## Overview

AI Football Analysis is a computer vision system that processes football match videos to automatically track players, detect the ball, and analyze play dynamics. Built using YOLOv5 and ByteTrack, the system generates annotated videos showing player positions, team assignments, ball possession, and movement analytics.

## Features

- **Custom Object Detection**: Trained YOLOv5 model to identify players, referees, goalkeepers, and the ball.
- **Player Detection & Tracking**: Identifies and tracks players with team classification.
- **Ball Detection & Possession**: Tracks ball movement with position interpolation for continuous tracking.
- **Team Identification**: Classifies players by team using jersey color clustering.
- **Camera Movement Estimation**: Adjusts tracking for dynamic camera footage.
- **Speed & Distance Estimation**: Calculates and displays player movement statistics.
- **Annotation System**: Visualizes tracks, team assignments, and possession data.

## Technologies Used

- **Python**
- **YOLOv5** (Object Detection)
- **ByteTrack** (Object Tracking)
- **OpenCV**
- **NumPy**

## Getting Started

To get started with AI Football Analysis, clone the repository and follow the setup instructions.

```sh
git clone https://github.com/abrarshahok/AI-Football-Analysis.git
cd AI-Football-Analysis
```
