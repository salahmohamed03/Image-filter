# Message Documentation for MainWindowUI

## Subscribed Messages

| Message Name   | Handler         | Description                                           |
|---------------|----------------|-------------------------------------------------------|
| "update display" | update_display() | Updates the UI display when image processing operations complete |

## Published Messages

| Message Name            | Parameters              | Description                                          |
|------------------------|------------------------|------------------------------------------------------|
| "Add Noise"            | noise: Noise           | Sends noise parameters for adding noise to image    |
| "Edge Detection"       | filter: str            | Triggers edge detection with specified filter type  |
| "Mix Images"          | freq1: int, freq2: int | Sends frequency cutoff values for image mixing      |
| "Normalize Image"      | None                   | Triggers image normalization operation              |
| "Histogram Equalization" | None                 | Triggers histogram equalization operation           |
| "Thresholding"         | None                   | Triggers thresholding operation                     |
