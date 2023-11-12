#!/usr/bin/env python

import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Union
from urllib.parse import urlparse

import click
import cv2

from src.misc import clean_str

logger = logging.getLogger(__name__)


class StreamLoader:
    """Streamloader, i.e. `# RTSP, RTMP, HTTP streams`."""

    def __init__(
        self,
        sources: Union[str, list] = "file.streams",
        buffer: bool = True,
        buffer_length: Union[str, int] = "auto",
        vid_fps: Union[str, int] = "auto",
    ) -> "StreamLoader":
        """Initialize instance variables and check for consistent input stream shapes.
        Args:
            sources: File with streams or a list of stream links
            buffer: Keep `buffer_length` last images for each stream or keep last one
            buffer_length: Length of buffer. If -1 - auto mode.
            vid_fps: New FPS for all videos. if 'auto', all videos will be aligned by min fps in streams.
                     If `isinstance(vid_fps, int)` - all videos will be set to this value.
                     If vid_fps == -1, fps alignment will not be executed.
        """
        # torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        # check input args
        for var_name in ("buffer_length", "vid_fps"):
            var_value = eval(var_name)
            assert (
                isinstance(var_value, str) and var_value == "auto"
            ) or (
                var_value == -1 or var_value > 0
            ), f"Set {var_name} > 0 or equal -1, got {var_value}."
        self.buffer = buffer  # buffer input streams
        self.buffer_length = buffer_length  # max buffer length
        self.running = True  # running flag for Thread
        self.mode = "stream"
        if type(sources) == str:
            sources = (
                Path(sources).read_text().rsplit()
                if os.path.isfile(sources)
                else [sources]
            )
        elif type(sources) == list:
            pass
        else:
            raise ValueError(f"Unsupported sources type: {sources}")
        n = len(sources)
        self.sources = [
            clean_str(x) for x in sources
        ]  # clean source names for later
        self.imgs = []  # buffer with images
        self.fps = [0] * n  # fps of each stream
        self.frames = [0] * n  # number of frames in each stream
        self.threads = [None] * n  # buffer stored streams
        self.shape = [[] for _ in range(n)]
        self.caps = [None] * n  # video capture objects
        self.buffer_lengths = [0] * n  # keep buffer lengths
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            if urlparse(s).hostname in (
                "www.youtube.com",
                "youtube.com",
                "youtu.be",
            ):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
                raise NotImplementedError(
                    f"Kaggle, YouTube are not supported now."
                )
            s = (
                eval(s) if s.isnumeric() else s
            )  # i.e. s = '0' local webcam
            self.caps[i] = cv2.VideoCapture(
                s
            )  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            # warning: may return 0 or nan
            if not math.isfinite(fps) or fps == 0:
                logger.warning(
                    f"{st}Warning ⚠️ Stream returned {fps} FPS, set defaut 30 FPS."
                )
            self.frames[i] = max(
                int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0
            ) or float(
                "inf"
            )  # infinite stream fallback
            self.fps[i] = (
                max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            )  # 30 FPS fallback
            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                raise ConnectionError(
                    f"{st}Failed to read images from {s}"
                )
            stream_buffer_length = (
                math.ceil(self.fps[i])
                if (
                    isinstance(buffer_length, str)
                    and buffer_length == "auto"
                )
                or buffer_length == -1
                else buffer_length
            )
            self.buffer_lengths[i] = stream_buffer_length
            buf = deque(maxlen=stream_buffer_length)
            buf.append(im)
            self.imgs.append(buf)
            self.shape[i] = im.shape
            self.threads[i] = Thread(
                target=self.update,
                args=([i, self.caps[i], s]),
                daemon=True,
            )
            logger.info(
                f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)"
            )

        self.new_fps = (
            min(self.fps)
            if isinstance(vid_fps, str) and vid_fps == "auto"
            else vid_fps
        )  # fps alignment
        logger.info("")  # newline

        # run all threads
        for i in range(n):
            self.threads[i].start()

        # Check for common shapes
        self.bs = self.__len__()

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            success = (
                cap.grab()
            )  # .read() = .grab() followed by .retrieve()
            if not success:
                im = None
                logger.warning(
                    f"WARNING ⚠️ Video stream {i} unresponsive, please check your IP camera connection."
                )
                cap.open(stream)  # re-open stream if signal was lost
            else:
                success, im = cap.retrieve()
                if not success:
                    im = None
                    logger.warning(
                        f"WARNING ⚠️ Cannot decode image from video stream {i}. Unknown error."
                    )
            self.imgs[i].append(im)
            n += 1
        else:
            logger.info(f"End of stream {i}.")

    def close(self):
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for (
            cap
        ) in (
            self.caps
        ):  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                logger.warning(
                    f"WARNING ⚠️ Could not release VideoCapture object: {e}"
                )
        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns original images for processing."""
        self.count += 1
        images = []

        # sleep to align fps
        time.sleep(1 / self.new_fps)

        for i, x in enumerate(self.imgs):
            # If image is not available
            if not x:
                if not self.threads[i].is_alive() or cv2.waitKey(
                    1
                ) == ord(
                    "q"
                ):  # q to quit
                    self.close()
                    raise StopIteration
                logger.warning(f"WARNING ⚠️ Waiting for stream {i}")
                im = None
            # Get the last element from buffer
            else:
                # Main process just read from buffer, not delete
                im = x[-1]

            images.append(im)

        return images

    def __len__(self):
        """Return the length of the sources object."""
        return len(
            self.sources
        )  # 1E12 frames = 32 streams at 30 FPS for 30 years


@click.command()
@click.option(
    "--streams_path",
    "-s",
    type=str,
    help="Path to streams.",
)
def main(
    streams_path: str,
) -> None:
    loader = LoadStreams(
        sources=streams_path,
        buffer=True,
    )

    click = time.perf_counter()

    for i, ims in enumerate(loader):
        clack = time.perf_counter()
        print(f"Time elapsed {clack - click} s")
        click = clack


if __name__ == "__main__":
    main()
