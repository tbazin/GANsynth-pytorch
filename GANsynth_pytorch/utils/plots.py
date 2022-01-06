import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import functools


def plot_mel_representations(log_melspec: np.ndarray, mel_IF: np.ndarray,
                             hop_length: int, fs_hz: int,
                             ax_spec=None, ax_IF=None,
                             print_title: bool = True,
                             **kwargs
                             ) -> None:
    ax = ax_spec or plt.subplot(1, 2, 1)
    librosa.display.specshow(log_melspec, sr=fs_hz, hop_length=hop_length,
                             ax=ax, **kwargs)
    if print_title:
        if ax_spec is not None:
            ax.set_title("log-melspectrogram")
        else:
            plt.title("log-melspectrogram")

    ax = ax_IF or plt.subplot(1, 2, 2)
    librosa.display.specshow(mel_IF, sr=fs_hz, hop_length=hop_length, ax=ax,
                             **kwargs)
    if print_title:
        if ax_IF is not None:
            ax.set_title("mel-IF")
        else:
            plt.title("mel-IF")

    # plt.tight_layout()

    # return plt.gcf()


def plot_mel_representations_batch(log_melspecs: np.ndarray,
                                   mel_IFs: np.ndarray,
                                   hop_length: int, fs_hz: int,
                                   **kwargs):
    # input data shape is [BATCH, FREQ, TIME]
    # with BATCH = 2*INPUT samples, containing originals and reconstructions
    num_samples = log_melspecs.shape[0] // 2
    #
    num_plots_per_row = 2*num_samples
    num_rows = 2

    make_plot = functools.partial(plot_mel_representations,
                                  hop_length=hop_length, fs_hz=fs_hz,
                                  print_title=False,
                                  **kwargs)

    fig, axes = plt.subplots(num_rows, num_plots_per_row,
                             figsize=(num_samples*5, 6))
    for sample_index in range(num_samples):
        for sample_type in range(0, 2):
            mel_spec = log_melspecs[sample_index + num_samples*sample_type]
            mel_IF = mel_IFs[sample_index + num_samples*sample_type]
            plot_position = sample_type*num_plots_per_row + 2*sample_index + 1
            ax_spec = plt.subplot(num_rows, num_plots_per_row, plot_position)
            ax_IF = plt.subplot(num_rows, num_plots_per_row, plot_position+1)
            make_plot(log_melspec=mel_spec, mel_IF=mel_IF, ax_spec=ax_spec,
                      ax_IF=ax_IF)

    return fig, axes


if __name__ == '__main__':
    import librosa
    n_synth_sample_duration_s = 4.0
    y, fs_hz = librosa.load(librosa.util.example_audio_file(),
                            duration=n_synth_sample_duration_s)
    hop_length = 512
    Y = librosa.stft(y, hop_length=hop_length)
    magnitude, phase = librosa.magphase(Y)
    D = librosa.amplitude_to_db(magnitude, ref=np.max)
    angle = np.angle(phase)
    mel_D = librosa.feature.melspectrogram(S=D)
    mel_angle = librosa.feature.melspectrogram(S=angle)

    num_samples = 3
    # replicate spec and phase to get `num_samples` samples
    # and repeat a second time to emulate original/reconstructions pairs
    mel_specs, mel_angles = [
        np.expand_dims(array, 0).repeat(2, axis=0).repeat(num_samples, axis=0)
        for array in [mel_D, mel_angle]]

    fig, axes = plot_mel_representations_batch(log_melspecs=mel_specs,
                                               mel_IFs=mel_angles,
                                               hop_length=hop_length,
                                               fs_hz=fs_hz)
    fig.tight_layout()
    plt.show()
