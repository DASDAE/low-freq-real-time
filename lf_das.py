"""
A Python script for low frequency processing on DAS data.
"""

import dascore as dc
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

from xarray.core.utils import FrozenDict


# define functions
def _check_merge(plist):
    if len(plist) > 1:
        raise Exception("patch merge failed! Gap in data exists")
    else:
        return plist[0]


def _get_timestr(bgtime: np.datetime64) -> str:
    timestr = str(bgtime.astype("datetime64[ms]"))[:21]
    timestr = timestr.replace(":", "")  # for windows compatable
    return timestr


def _get_filename(bgtime, edtime) -> str:
    filename = (
        "LFDAS_" + _get_timestr(bgtime) + "_" + _get_timestr(edtime) + ".h5"
    )  # changed by ahmad
    # + '_' + _get_timestr(edtime) + '.p'
    return filename


def _down_sample_processing(patch, freq=5, nqfreq_ratio=0.8, **kargs):
    dt = np.timedelta64(int(1 / freq * 1e9), "ns")
    corner_f = freq * 0.5 * nqfreq_ratio

    proc_patch = patch

    proc_patch = proc_patch.pass_filter(time=(None, corner_f))
    new_taxis = np.arange(patch.attrs["time_min"], patch.attrs["time_max"], dt)
    proc_patch = proc_patch.interpolate(time=new_taxis)

    return proc_patch


def get_edge_effect_time(
    sampling_interval, total_T, fun=_down_sample_processing, tol=1e-6, **kargs
):
    N = int(total_T / sampling_interval)

    taxis = (np.arange(N) - N // 2) * sampling_interval
    data = np.zeros_like(taxis)
    data[N // 2] = 1

    coords = {"time": dc.to_datetime64(taxis), "distance": [0]}
    data = data.reshape((-1, 1))
    attrs = {"d_time": sampling_interval, "d_distance": 1}

    newdata = dc.Patch(data=data, coords=coords, dims=["time", "distance"], attrs=attrs)
    process_data = newdata.pipe(fun, **kargs)

    data = process_data.data[:, 0]

    max_val = np.max(np.abs(data))
    ind = np.abs(data) > max_val * tol
    ind_1 = np.where(ind)[0][0]
    ind_2 = np.where(ind)[0][-1]

    new_taxis = process_data.coords["time"]
    new_taxis = (new_taxis - new_taxis[0]) / np.timedelta64(
        1, "s"
    ) - N // 2 * sampling_interval

    edge_t = max(np.abs(new_taxis[ind_1]), np.abs(new_taxis[ind_2]))

    if edge_t * 2 >= total_T:
        raise ValueError(
            f"edge_t {edge_t} s is equal or larger than half of\
            the processing chunk size {total_T} s.\
            Please increase memory_size or tolerance."
        )

    return edge_t


def get_patch_time(
    memory_size,
    sampling_rate,
    num_ch,
    bytes_per_element=8,
    processing_factor=5,
    memory_safety_factor=1.2,
):
    mem_size_per_second = (
        sampling_rate
        * num_ch
        * bytes_per_element
        * processing_factor
        * memory_safety_factor
        / 1e6
    )  # in MB
    patch_length = memory_size / mem_size_per_second  # in sec
    return patch_length


def calculate_mean_over_samples(data, step_size=1000, axis=0):
    total_samples = data.shape[0]
    mean_values = np.empty((int(total_samples / step_size), data.shape[1]))
    for j, k in enumerate(range(0, total_samples, step_size)):
        mean_values[j, :] = np.mean(data[k : k + step_size, :], axis=axis)
    return mean_values


def waterfall_plot(
    some_data,
    min_sec,
    max_sec,
    min_ch,
    max_ch,
    ch_start,
    channel_spacing,
    surface_fiber,
    sample_rate,
    fig_title,
    fig_dir,
    fig_name,
):
    # Basic error checking
    if (
        (min_sec >= max_sec)
        or (min_sec < 0)
        or (max_sec * sample_rate > some_data.shape[1])
    ):
        print(
            "ERROR in plotSpaceTime inputs minSec: "
            + str(min_sec)
            + " or maxSec: "
            + str(max_sec)
        )
        return
    if (min_ch >= max_ch) or (min_ch < 0) or (max_ch > some_data.shape[0]):
        print(
            "Error in plotSpaceTime inputs minCh: "
            + str(min_ch)
            + " or maxCh: "
            + str(max_ch)
            + " referring to array with "
            + str(some_data.shape[0])
            + " channels."
        )
        return

    # turn time range (in seconds) to indices
    minSecID = int(min_sec * sample_rate)
    maxSecID = int(max_sec * sample_rate)

    # to get reasonable saturation, clip the values
    perc_clip = 95
    clip_val = np.percentile(np.absolute(some_data), perc_clip)

    # make the plot
    plt.figure(figsize=(12, 8))
    plt.imshow(
        some_data[min_ch:max_ch, minSecID:maxSecID],
        aspect="auto",
        interpolation="none",
        cmap="seismic",
        extent=(
            min_sec,
            max_sec,
            (max_ch + ch_start) * channel_spacing - surface_fiber,
            (min_ch + ch_start) * channel_spacing - surface_fiber,
        ),
        vmin=-clip_val,
        vmax=clip_val,
    )
    plt.ylabel("MD (ft)", fontsize=10)
    plt.xlabel("Time (sec)", fontsize=10)
    plt.title(fig_title, fontsize=14)
    plt.colorbar().set_label("Strain rate (1/s)", fontsize=10)
    plt.savefig(fig_dir + "/" + fig_name + ".jpeg", dpi=600, format="jpeg")
    plt.show()


# define the main class
class LFProc:
    def __init__(self, sp=None):
        self._spool = sp
        self._para = self._default_process_parameters()
        self._output_folder = None

    def set_output_folder(self, folder, delete_existing=False):
        self._output_folder = folder
        if delete_existing and os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"original {folder} deleted")
        if not os.path.isdir(folder):
            os.mkdir(folder)
            print(f"{folder} created")

    def _default_process_parameters(self):
        para = {
            "output_sample_interval": 1.0,  # in seconds
            "process_patch_size": 100,  # in number of patches \
            # (Each patch will be in size of the output sample interval sec.)
            "edge_buff_size": 10,  # in number of patches \
            # (Each patch will be in size of the output sample interval sec.)
            "data_gap_tolorance": 10.0,
        }
        return para

    def update_processing_parameter(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self._para.keys():
                print(f"{key} is not default parameter key")
            else:
                self._para[key] = value
        return self.parameters

    def get_last_processed_time(self):
        out_sp = dc.spool(self._output_folder).update()
        t_last = out_sp[-1].attrs["time_max"]
        return t_last

    def process_time_range(self, bgtime, edtime):
        # define the main processing flow
        def lp_process(DASdata, bgind, edind):
            # low pass filter and downsampling
            lfDAS = DASdata.pass_filter(time=(None, 1 / dt / 2 * 0.9)).interpolate(
                time=time_grid[bgind:edind]
            )

            lfDAS = lfDAS.update_attrs(d_time=dt)

            # output the result to output folder
            filename = _get_filename(lfDAS.attrs["time_min"], lfDAS.attrs["time_max"])
            filename = self._output_folder + "/" + filename
            lfDAS.io.write(filename, "dasdae")

        # define the processing flow to avoid repeat code
        def merge_and_process(DASdata):
            DASdata = self._spool.select(
                time=(time_grid[data_end - 2 * buff_size], time_grid[new_data_end])
            )
            plist = dc.spool(DASdata).chunk(time=None)
            DASdata = _check_merge(plist)

            # low pass filter and down sample
            lp_process(DASdata, data_end - buff_size, new_data_end - buff_size)
            return DASdata

        if self._output_folder is None:
            raise Exception("Please setup output folder first")
        dt = self._para["output_sample_interval"]
        patch_size = self._para["process_patch_size"]
        buff_size = self._para["edge_buff_size"]

        time_grid = np.arange(
            bgtime.astype("datetime64[ns]"),
            edtime.astype("datetime64[ns]"),
            np.timedelta64(int(dt * 1000), "ms"),
        )

        if len(time_grid) <= patch_size:
            patch_size = len(time_grid) - 1

        # load and process the first patch
        i = 1
        print("Processing patch ", str(i))
        plist = self._spool.select(time=(time_grid[0], time_grid[patch_size]))
        plist = dc.spool(plist).chunk(time=None)
        DASdata = _check_merge(plist)

        # low pass filter and downsampling
        lp_process(DASdata, buff_size, patch_size - buff_size)

        # processing for the rest of the dataset
        data_end = patch_size
        new_data_end = data_end + patch_size - 2 * buff_size
        while new_data_end < len(time_grid):
            i += 1
            print("Processing patch ", str(i))

            # reading new data
            DASdata = merge_and_process(DASdata)

            # update index
            data_end = new_data_end
            new_data_end = data_end + patch_size - 2 * buff_size

        # dealing with the rest of data smaller than patch_size
        if (len(time_grid) - data_end) > 1:
            i += 1
            new_data_end = len(time_grid) - 1
            DASdata = merge_and_process(DASdata)
            print(f"Processing patch {i} (Last portion of data)")

    ### property definiations
    @property
    def parameters(self):
        return FrozenDict(self._para)
