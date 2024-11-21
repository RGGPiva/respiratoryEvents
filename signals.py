import pathlib as pl

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib import pyplot as plt
from scipy.ndimage import label, uniform_filter1d, median_filter
from tqdm.contrib.concurrent import process_map


def percentile_threshold(signal: np.ndarray, percentile: int = 15) -> float:
  """Calculate a threshold value for a signal based on its percentile.

  Parameters
  ----------
  signal : np.ndarray
      The input signal array.
  percentile : int, optional
      The percentile to use to calculate the threshold. Default is 15.

  Returns:
  -------
  threshold : float
      The threshold value.
  """
  return np.percentile(np.abs(signal), percentile)


def dynamic_range_threshold(signal: np.ndarray, fraction: float = 0.05) -> float:
  """Calculate a threshold value for a signal based on its dynamic range.

  Parameters
  ----------
  signal : np.ndarray
      The input signal array.
  fraction : float, optional
      The fraction of the dynamic range to use to calculate the threshold. Default is 0.05.

  Returns:
  -------
  threshold : float
      The threshold value.
  """
  dynamic_range = np.max(signal) - np.min(signal)
  return dynamic_range * fraction


def signal_noise_level_threshold(signal: np.ndarray, factor: float = 0.2) -> float:
  """Calculate a threshold value for a signal based on its standard deviation.

  Parameters
  ----------
  signal : np.ndarray
      The input signal array.
  factor : float, optional
      The factor to multiply the standard deviation by to calculate the threshold. Default is 0.2.

  Returns:
  -------
  threshold : float
      The threshold value.
  """
  return np.std(signal) * factor


def mad_threshold(signal: np.ndarray, scale_factor: float = 1.4826) -> float:
  """Calculate a threshold value for a signal based on its Median Absolute Deviation (MAD).

  Parameters
  ----------
  signal : np.ndarray
      The input signal array.
  scale_factor : float, optional
      The factor to multiply the MAD by to calculate the threshold. Default is 1.4826.

  Returns:
  -------
  threshold : float
      The threshold value.
  """
  median = np.median(signal)
  mad = np.median(np.abs(signal - median))
  return mad * scale_factor


def process_signal(idx: int, signal: np.ndarray, plots_dir: pl.Path) -> None:
  """Process a signal by smoothing it, identifying dead intervals, and generating a plot.

  This function performs the following operations on the input signal:
  1. Smooths the signal using a uniform filter.
  2. Calculates the dynamic range and defines a threshold based on the range.
  3. Identifies dead intervals in the signal where values fall below the threshold.
  4. Plots the original signal, smoothed signal, threshold, and dead intervals.
  5. Saves the resulting plot as a PNG file in the specified directory.

  Parameters
  ----------
  idx : int
      The index of the signal in the dataset. Used to name the output plot file.
  signal : np.ndarray
      The raw signal data to be processed.
  plots_dir : pl.Path
      The directory where the plot will be saved.

  Returns:
  -------
  None
      This function does not return any value. It saves the plot as a PNG file.

  Notes:
  -----
  The function applies a smoothing filter to the signal, identifies intervals where the
  signal is below a calculated threshold (considered as "dead intervals"), and visualizes
  the original and smoothed signals along with the detected dead intervals.
  """
  # Skip signal if it is empty
  if np.abs(signal).sum() == 0:
    return
  # Create a new figure for the plot
  plt.figure(figsize=(10, 7))

  # Step 1: Smooth the signal using a uniform filter
  smoothed_signal = uniform_filter1d(signal, size=200)  # Smoothing with a window size of 200

  # Step 2: Calculate dynamic range and define threshold
  threshold = dynamic_range_threshold(smoothed_signal)

  # Step 3: Identify dead points (points below the threshold)
  dead_points = np.abs(smoothed_signal) < threshold

  # Step 4: Label connected dead points and find intervals
  labeled_array, num_features = label(dead_points)
  intervals = []
  dead_signal_length = round(len(signal) * 0.05)  # Minimum length for a dead interval (5% of signal)

  # Iterate through each feature (dead interval) and store its start and end points
  for i in range(1, num_features + 1):
    indices = np.where(labeled_array == i)[0]
    if len(indices) >= dead_signal_length:  # Only consider intervals longer than the threshold
      intervals.append((indices[0], indices[-1]))

  # Step 5: Merge intervals that are close together
  min_gap = int(0.02 * len(signal))  # 3% of the signal length
  merged_intervals = []
  current_interval = None
  for interval in intervals:
    if current_interval is None:
      current_interval = interval
    elif interval[0] - current_interval[1] < min_gap:
      current_interval = (current_interval[0], interval[1])
    else:
      merged_intervals.append(current_interval)
      current_interval = interval
  if current_interval:
    merged_intervals.append(current_interval)

  # Step 6: Plot the signal, smoothed signal, and threshold lines
  plt.plot(signal, label="Signal")
  plt.plot(smoothed_signal, label="Smoothed signal")
  plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
  plt.axhline(y=-threshold, color="r", linestyle="--")

  # Add vertical spans for each dead interval
  for start, end in merged_intervals:
    plt.axvspan(
      start - 0.5,
      end + 0.5,
      color="gray",
      alpha=0.3,
      label="Dead interval" if "Dead interval" not in plt.gca().get_legend_handles_labels()[1] else None,
    )

  # Step 7: Add title, labels, and legend
  plt.title(f"Dead ranges in signal (n = {len(merged_intervals)}, threshold={threshold:.2f})")
  plt.xlabel("Time")
  plt.ylabel("Amplitude")
  plt.legend()

  # Step 8: Finalize plot layout and save the figure
  plt.tight_layout()
  plt.savefig(plots_dir / f"{idx:03d}.png")
  plt.savefig(plots_dir / f"{idx:03d}.pdf")
  plt.close()


def process_signal_wrapper(args):
  """Wrapper function to unpack arguments and call process_signal.

  Parameters
  ----------
  args : tuple
      A tuple containing the arguments to be passed to process_signal.

  Returns:
  --------
  None
      This function does not return any value. It calls process_signal with
      unpacked arguments.
  """
  return process_signal(*args)


def main() -> None:
  plots_dir = pl.Path("plots/")
  plots_dir.mkdir(exist_ok=True)

  txt_data = pl.Path("events20mb.txt").read_text(encoding="utf-8").split("\n")
  txt_data = [np.array(list(map(int, x.split(",")[1:])), dtype=np.int32) for x in txt_data if x]
  lens = [len(x) for x in txt_data]
  print(
    f"Number of signals: {len(txt_data)}\nSmallest: {min(lens)}\nLargest: {max(lens)}\nMean size: {np.mean(lens)}\nMedian size: {np.median(lens)}"
  )

  # Histogram
  plt.figure(figsize=(12, 7))
  _, _, bars = plt.hist(lens, bins=25)
  plt.bar_label(bars, padding=3)
  plt.xlabel("Signal length")
  plt.ylabel("Number of signals")
  plt.tight_layout()
  plt.savefig(plots_dir / "histogram.png")
  plt.close()

  # Process signals
  process_map(
    process_signal_wrapper,
    [(idx, signal, plots_dir) for idx, signal in enumerate(txt_data) if np.abs(signal).sum() > 0],
    max_workers=8,
  )


if __name__ == "__main__":
  main()
