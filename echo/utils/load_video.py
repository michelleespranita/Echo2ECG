import numpy as np
import os
from decord import VideoReader, cpu

def loadvideo_decord(
    fname,
    mode='val',
    frame_sample_rate=4,
    clip_len=16,
    must_include_frame_ids=[],
    repeat_or_pad='repeat'
):

    # load videoreader
    if not (os.path.exists(fname)):
        return []

    # avoid hanging issue
    if os.path.getsize(fname) < 1 * 1024:
        print('SKIP: ', fname, " - ", os.path.getsize(fname))
        return []
    try:
        vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    except:
        print("video cannot be loaded by decord: ", fname)
        return []
    
    repeat_or_pad_video_ids = [] # to store the repeated/padded indices in the final video, for attention masks
    # select frame indices to load from video
    if frame_sample_rate == 'uniform':
        if clip_len > len(vr): 
            frame_id_list = [x for x in range(0, len(vr))]
            if repeat_or_pad == 'repeat': # Repeat the last frame if the video is shorter than clip_len
                i = len(frame_id_list)
                while len(frame_id_list) < clip_len:
                    frame_id_list.append(frame_id_list[-1])
                    repeat_or_pad_video_ids.append(i)
                    i += 1
        else: # Uniform sampling
            frame_id_list = np.linspace(0, len(vr) - 1, num=clip_len, dtype=int)

            for fid in must_include_frame_ids:
                if fid in frame_id_list:
                    continue
                closest_idx = np.argmin(np.abs(frame_id_list - fid))
                frame_id_list[closest_idx] = fid

            frame_id_list = frame_id_list.tolist()

    else:
        if not mode == 'train': # buffer has to have at least clip_len frames, but is allowed to have more. sampling_rate is applied.
            frame_id_list = [x for x in range(0, len(vr), frame_sample_rate)]
            if repeat_or_pad == 'repeat':
                i = len(frame_id_list)
                while len(frame_id_list) < clip_len:
                    frame_id_list.append(frame_id_list[-1])
                    repeat_or_pad_video_ids.append(i)
                    i += 1
            frame_id_list = frame_id_list[:clip_len] # this forces buffer to have clip_len frames
        
        else: # buffer has to have exactly clip_len frames. sampling_rate is applied.
            converted_len = int(clip_len * frame_sample_rate)
            seg_len = len(vr)

            frame_id_list = []
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // frame_sample_rate)
                if repeat_or_pad == 'repeat':
                    repeat_or_pad_video_ids = [len(index) + i for i in range(clip_len - seg_len // frame_sample_rate)]
                    index = np.concatenate((index, np.ones(clip_len - seg_len // frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            
            index = list(map(int, index))
            frame_id_list.extend(index)
        
    # load video
    vr.seek(0)
    buffer = vr.get_batch(frame_id_list).asnumpy()

    # pad video if necessary
    if repeat_or_pad == 'pad':
        T, C, H, W = buffer.shape
        if T < clip_len:
            missing_num_frames = clip_len - T
            padding = np.zeros((missing_num_frames, C, H, W)).astype(buffer.dtype)
            buffer = np.concatenate((buffer, padding), axis=0)
            repeat_or_pad_video_ids = [len(frame_id_list) + i for i in range(missing_num_frames)]
    
    return {
        'video': buffer,
        'frame_id': frame_id_list,
        'repeat_or_pad_video_id': repeat_or_pad_video_ids,
        'num_repeat_or_pad_frames': len(repeat_or_pad_video_ids)
    }
    
def loadvideo_decord_full(sample):
    """
    Output: numpy array (T, H, W, C)
    """
    fname = sample

    if not (os.path.exists(fname)):
        return []

    # avoid hanging issue
    if os.path.getsize(fname) < 1 * 1024:
        print('SKIP: ', fname, " - ", os.path.getsize(fname))
        return []
    try:
        vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    except:
        print("video cannot be loaded by decord: ", fname)
        return []

    vr.seek(0)
    frame_id_list = list(range(len(vr)))
    buffer = vr.get_batch(frame_id_list).asnumpy()

    return {
        'video': buffer,
        'frame_id': frame_id_list
    }



