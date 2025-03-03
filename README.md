# AI Football Analysis

## Overview

AI Football Analysis is a computer vision-based system that automatically processes football match videos to track players, detect the ball, and analyze team possession. The system leverages **YOLOv5** for object detection and **ByteTrack** for tracking, generating annotated videos that display player positions, team assignments, ball possession, and camera movements.

## Features

- **Player Detection & Tracking**: Identifies and tracks players with team classification.
- **Ball Detection**: Detects the ball and interpolates its position across frames.
- **Team Identification**: Uses jersey color clustering to distinguish teams.
- **Ball Possession Analysis**: Determines and visualizes which team has possession.
- **Camera Movement Estimation**: Handles moving camera footage for stable tracking.
- **Annotation System**: Displays player tracks, team assignments, and possession overlays.
- **Pre-Computed Detection Support**: Allows faster processing by utilizing pre-detected data.

## Getting Started

To get started with AI Football Analysis, clone the repository and follow the setup instructions.

```sh
git clone https://github.com/abrarshahok/AI-Football-Analysis.git
cd AI-Football-Analysis
```
