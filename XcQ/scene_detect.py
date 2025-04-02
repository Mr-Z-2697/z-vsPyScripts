'''https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/main/src/scene_detect.py
Model Download: https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models
(look for onnx models start with "sc_")

USAGE:
import scene_detect as scd
clip=scd.scene_detect(clip,[model path, [other parameters]])

BSD 3-Clause License

Copyright (c) 2022, styler00dollar aka sudo rm -rf / --no-preserve-root
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'''

import os
import numpy as np
import vapoursynth as vs
import functools
import onnxruntime as ort
from threading import Lock
core = vs.core

def scene_detect(
    clip: vs.VideoNode,
    onnx_path: str = r"D:\Misc\scd-models\maxxvitv2_nano_rw_256.sw_in1k_256px_b100_30k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx",
    thresh: float = 0.92,
    fp16=None,
    onnx_res=None,
    onnx_type=None,
    ort_provider='Dml',
    resizer=None,
    model_rev=None,
    output_type=None,
    ssim_clip=None,
    ssim_thresh=0.98,
    num_sessions=1,
    ort_log_level=3,
    return_clip=None,
) -> vs.VideoNode:
    
    if not isinstance(return_clip,vs.VideoNode):
        return_clip=clip
    
    ort.set_default_logger_severity(ort_log_level)
    
    onnx_name=os.path.split(onnx_path)[-1]
    onnx_name=os.path.splitext(onnx_name)[0]
    onnx_name_split=onnx_name.split('_')

    if fp16 is None:
        if 'fp16' in onnx_name_split:
            fp16=True
        else:
            fp16=False

    if onnx_res is None:
        if '256' in onnx_name_split or '256px' in onnx_name_split:
            onnx_res=[256,256]
        elif '224' in onnx_name_split or '224px' in onnx_name_split:
            onnx_res=[224,224]
        elif 'autoshot' in onnx_name_split or '48x27' in onnx_name_split:
            onnx_res=[48,27]
    elif isinstance(onnx_res,int):
        onnx_res=[onnx_res,onnx_res]

    if onnx_type is None:
        if 'autoshot' in onnx_name_split or '5img' in onnx_name_split:
            onnx_type='5img'
        else:
            onnx_type='2img'

    if model_rev is None:
        if 'softmaxOut' in onnx_name_split:
            model_rev=1
        else:
            model_rev=2

    if output_type is None:
        if 'dists' in onnx_name_split:
            output_type=2
        else:
            output_type=1

    options = {}
    '''
    I can't test these options because I can't get TRT provider to work in my Windows system.
    The "trt_engine_cache_path" looks definitely gonna cause some trouble (in Windows, perhaps Linux as well, as the upstream is intended for docker).
    '''
    if ort_provider=='Tensorrt':
        # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
        options["device_id"] = 0
        options["trt_engine_cache_enable"] = True
        options["trt_timing_cache_enable"] = (
            True  # Using TensorRT timing cache to accelerate engine build time on a device with the same compute capability
        )
        options["trt_engine_cache_path"] = (
            "/workspace/tensorrt/"
        )
        options["trt_fp16_enable"] = fp16
        options["trt_max_workspace_size"] = 7000000000  # ~7gb
        options["trt_builder_optimization_level"] = 5

    sessions = [
        ort.InferenceSession(
            onnx_path,
            providers=[(f"{ort_provider}ExecutionProvider",options)],
        )
        for _ in range(num_sessions)
    ]
    sessions_lock = [Lock() for _ in range(num_sessions)]

    index=-1
    index_lock=Lock()

    def execute(n,f):
        nonlocal index
        with index_lock:
            index = (index + 1) % num_sessions
            local_index = index
        nonlocal ssim_clip
        nonlocal ssim_thresh
        if ssim_clip:
            ssim_eval = f[3].props.get("float_ssim")
            if ssim_clip and ssim_eval > ssim_thresh:
                return f[0].copy()
        fout=f[0].copy()
        I0 = frame_to_tensor(f[1])
        I1 = frame_to_tensor(f[2])
        ort_session = sessions[local_index]
        if model_rev==1:
            I0 = np.expand_dims(I0, 0)
            I1 = np.expand_dims(I1, 0)
            in_sess = np.concatenate([I0, I1], axis=1)
            result = ort_session.run(None, {"input": in_sess})[0]
        elif model_rev==2:
            if onnx_type=='2img':
                in_sess = np.concatenate([I0, I1], axis=0)
            elif onnx_type=='5img':
                I2 = frame_to_tensor(f[3])
                I3 = frame_to_tensor(f[4])
                I4 = frame_to_tensor(f[5])
                in_sess = np.stack([I0, I1, I2, I3, I4], axis=1)
            if output_type==2:
                result=ort_session.run(None, {"input": in_sess})[0]
            else:
                result = ort_session.run(None, {"input": in_sess})[0][0]
                if onnx_type=='2img':
                    result=result[0]
                elif onnx_type=='5img':
                    result=result[2]

        if result > thresh:
            fout.props._SceneChangeNext=1
        else:
            fout.props._SceneChangeNext=0
        fout.props._SceneChangeMetrics=float(result)
        return fout
    
    clip_down=prepare_clip(clip,onnx_res=onnx_res,fp16=fp16,resizer=resizer)
    if onnx_type == "5img":
        shift_up2 = clip_down.std.DeleteFrames(frames=[0, 1]) + core.std.BlankClip(
            clip_down, length=2
        )
        shift_up1 = clip_down.std.DeleteFrames(frames=[0]) + core.std.BlankClip(
            clip_down, length=1
        )

        shift_down1 = core.std.BlankClip(clip_down, length=1) + core.std.BlankClip(
            clip_down, length=1
        )
        shift_down2 = core.std.BlankClip(clip_down, length=2) + core.std.BlankClip(
            clip_down, length=2
        )

        return core.std.ModifyFrame(
            return_clip,
            (return_clip, shift_down2, shift_down1, clip_down, shift_up1, shift_up2),
            execute,
        )

    if ssim_clip:
        return core.std.ModifyFrame(return_clip,(return_clip,clip_down,clip_down[1:],ssim_clip),execute)
    return core.std.ModifyFrame(return_clip,(return_clip,clip_down,clip_down[1:]),execute)


def frame_to_tensor(frame: vs.VideoFrame):
    return np.stack(
        [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
    )
def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])
    return f
def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
    clip = core.std.BlankClip(
        clip=clip, width=image.shape[-1], height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image),
    )
def box_resizer(clip,w,h,*args,**kwargs):
    clip=core.fmtc.resample(clip,w,h,kernel='box',css='444')
    clip=core.resize.Point(clip,*args,**kwargs)
    return clip
def prepare_clip(clip,onnx_res,fp16,resizer):
    col_fam=clip.format.color_family
    target_fmt=[vs.RGBS,vs.RGBH][fp16]
    if not callable(resizer):
        if not hasattr(core,'fmtc'):
            resizer=functools.partial(core.resize.Bicubic,filter_param_a=-0.3,filter_param_b=0.15)
        else:
            resizer=box_resizer
    resizer=functools.partial(resizer,format=target_fmt)
    if clip.format.id==target_fmt and clip.width==onnx_res[0] and clip.height==onnx_res[1]:
        return clip
    if col_fam==vs.RGB:
        return resizer(clip,onnx_res[0],onnx_res[1])
    else:
        matrix=clip.get_frame(0).props.get("_Matrix")
        if matrix is None or matrix==2:
            if clip.width>=1280 or clip.height>=720:
                matrix=1
            else:
                matrix=5
        return resizer(clip,onnx_res[0],onnx_res[1],matrix_in=matrix)

def sc_copy(clip,propclip): # in case you need it, e.g. you want to preprocess the detection clip but not the result clip
    def execute(n,f):
        fout=f[0].copy()
        fout.props._SceneChangeNext=f[1].props._SceneChangeNext
        fout.props._SceneChangeMetrics=f[1].props._SceneChangeMetrics
        return fout
    return core.std.ModifyFrame(clip,[clip,propclip],execute)
