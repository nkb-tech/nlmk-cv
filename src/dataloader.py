#!/usr/bin/env python

from typing import Union, List

import click
import math
import os
import time
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import logging

import cv2
import numpy as np

from utils import clean_str


# set up logging to file
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in ' \
           'function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class LoadStreams:
    """Streamloader, i.e. `# RTSP, RTMP, HTTP streams`."""

    def __init__(
        self,
        sources: str = 'file.streams',
        vid_fps: Union[str, int] = 'auto',
        buffer: bool = True,
        buffer_length: int = 30,
    ) -> "LoadStreams":
        """Initialize instance variables and check for consistent input stream shapes.
        Args:
            sources: File with streams.
            vid_fps: New FPS for all videos. if 'auto', all videos will be aligned by min fps in streams.
                     If `isinstance(vid_fps, int)` - all videos will be set to this value.
                     If vid_fps == -1, fps alignment will not be executed.
            buffer: Keep `buffer_length` last images for each stream or keep last one
            buffer_length: Length of buffer
        """
        # torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.buffer_length = buffer_length  # max buffer length
        self.running = True  # running flag for Thread
        self.mode = 'stream'
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs = self.allocate_buffer(n)  # buffer with images
        self.fps = [0] * n  # fps of each stream
        self.frames = [0] * n  # number of frames in each stream
        self.threads = [None] * n  # buffer stored streams
        self.shape = self.allocate_buffer(n)
        self.caps = [None] * n  # video capture objects
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
                raise NotImplementedError(f'Kaggle, YouTube are not supported now.')
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f'{st}Failed to open {s}')
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                'inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                raise ConnectionError(f'{st}Failed to read images from {s}')
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            logger.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')

        # fps alignment
        self.delay_stemps = [0] * n
        min_fps = min(self.fps)  # min_fps cannot be zero
        if isinstance(vid_fps, str):
            if vid_fps == 'auto':
                self.new_fps = min_fps  # min_fps cannot be zero
                if self.all_fps_are_equal(self.fps, min_fps):
                    logger.info(f'All streams have the same FPS {self.new_fps}, alignment not needed in mode `auto`.')
                    self.fps_alignment = False
                else:
                    self.fps_alignment = True
            else:
                raise ValueError(f'Only auto policy is supported as string, got {vid_fps}')
        elif isinstance(vid_fps, (int, float)):
            if vid_fps == -1:
                self.fps_alignment = False
            elif vid_fps <= min_fps:
                self.new_fps = vid_fps
                if self.all_fps_are_equal(self.fps, vid_fps):
                    logger.info(f'All streams have the same FPS as vid_fps={vid_fps}.')
                    self.fps_alignment = False
                else:
                    self.fps_alignment = True
            else:
                raise ValueError(
                    f"`vid_fps` is {vid_fps}, but minimum FPS is {min_fps}. "
                    f"You cannot put `vid_fps` more than minimum FPS of all streams."
                )
        else:
            raise ValueError(f"Inputs are not supported.")

        if self.fps_alignment:
            logger.info(f'FPS alignment is ON with new FPS {self.new_fps} ✅')
        else:
            logger.info(f'FPS alignment is OFF ❌')

        logger.info('')  # newline

        # run all threads
        for i in range(n):
            self.threads[i].start()
        
        # Check for common shapes
        self.bs = self.__len__()

    @staticmethod
    def allocate_buffer(n: int) -> List[List]:
        """Allocates list of empty lists n elements."""
        return [[] for _ in range(n)]
    
    @staticmethod
    def all_fps_are_equal(input_list: List[int], input_value: int) -> bool:
        """Checks that all values in `input_list` are equal `input_value`."""
        for list_value in input_list:
            if input_value != list_value:
                return False

        return True

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < self.buffer_length:  # keep a <=30-image buffer
                cap.grab()  # .read() = .grab() followed by .retrieve()
                choose_frames = math.floor(n * self.new_fps / self.fps[i]) == self.delay_stemps[i] if self.fps_alignment else True
                if choose_frames:  # fps alignment
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        logger.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                        cap.open(stream)  # re-open stream if signal was lost
                    self.delay_stemps[i] += 1  # increment `delay_stemps`
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
                n += 1
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                logger.warning(f'WARNING ⚠️ Could not release VideoCapture object: {e}')
        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns original images for processing."""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            # Wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    logger.warning(f'WARNING ⚠️ Waiting for stream {i}')

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return images

    def __len__(self):
        """Return the length of the sources object."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


@click.command()
@click.option(
    '--streams_path',
    '-s',
    default='../example.streams',
    type=str,
    help='Path to streams.',
)
@click.option(
    '--output_image_path',
    '-o',
    default='../outputs',
    type=str,
    help='Path to save images from streams.',
)
def main(
    streams_path: str,
    output_image_path: str,
) -> None:
    
    import os
    import time

    loader = LoadStreams(
        sources=streams_path,
        buffer=True,
    )

    os.makedirs(output_image_path, exist_ok=True)

    click = time.perf_counter()

    for i, ims in enumerate(loader):
        clack = time.perf_counter()
        print(f'Time elapsed {clack - click} s')
        click = clack
        cv2.imwrite(os.path.join(output_image_path, f'im_{i}.jpg'), ims[0])


if __name__ == '__main__':
    main()
