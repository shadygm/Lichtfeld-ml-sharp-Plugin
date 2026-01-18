# SHARP 4D Video Plugin for LichtFeld Studio

<p align="center">
  <img src="assets/milo.gif" alt="SHARP 4D Video Plugin Demo" width="85%"/>
</p>


A 4D Gaussian Splatting plugin for LichtFeld Studio, powered by [SHARP](https://github.com/apple/ml-sharp). This tool enables the conversion of standard video files into 4D Gaussian Splat sequences and provides a dedicated player for 4D visualization within the LichtFeld environment.

## Features

- **Video-to-4DGS Conversion**: Automatically process video files into per-frame 3D Gaussian Splat (PLY) sequences using the SHARP model.
- **Integrated Playback**: dedicated "Sharp 4D Video" side panel for loading and playing back 4D sequences.
- **In-Process Inference**: Runs the SHARP inference pipeline directly within LichtFeld Studio's python environment (requires CUDA).

## Installation

### From GitHub (LichtFeld Studio v0.5+)


In LichtFeld Studio:
1. Open the **Plugins** panel.
2. Enter: `https://github.com/shadygm/Lichtfeld-ml-sharp-Plugin`
3. Click **Install**.

### Manual Installation

1. Clone this repository into your LichtFeld Studio plugins directory:
   ```bash
   cd ~/.lichtfeld/plugins
   git clone https://github.com/shadygm/Lichtfeld-ml-sharp-Plugin.git
   ```
2. Restart LichtFeld Studio.
3. The plugin will automatically create a virtual environment and install all dependencies (including `ml-sharp` requirements) upon first load or inspection.

## Usage

### 1. Open the Panel
Locate the **Sharp 4D Video** tab in the side panel area of LichtFeld Studio.

### 2. Processing a Video
1. Check **Input is Video File**.
2. Select your source video path using the file picker.
3. Click **Process Video**.
   - *Note: Processing can take significant time depending on video length and GPU power.*
   - Progress is displayed in the panel.
   - Output PLY files are saved in a folder named `<video_name>_gaussians` next to your input video.

### 3. Playing a Sequence
1. Once processing is complete (or if you have an existing sequence), uncheck **Input is Video File** (or it will auto-switch after processing).
2. Ensure the path points to the directory containing the `.ply` sequence.
3. Click **Load PLY Sequence**.
4. Use the playback controls:
   - **Play/Pause**: Toggle playback.
   - **Frame Slider**: Scrub through the timeline.
   - **FPS**: Adjust playback speed.

## Requirements

- **LichtFeld Studio**
- **NVIDIA GPU** with CUDA support (required for SHARP inference).
- **FFmpeg** (usually handled by the plugin's dependencies).

## Credits & License

This plugin integrates the **SHARP** (Spatio-temporal Hierarchical Auto-Regressive Point-clouds) architecture.

- **Plugin Code**: Released under [GPL-3.0-or-later](LICENSE).
- **SHARP Library**: Included as a library in `ml-sharp`. Please refer to `ml-sharp/LICENSE` and `ml-sharp/LICENSE_MODEL` for specific usage rights regarding the model and inference code.